// References:
// https://github.com/KhronosGroup/Vulkan-Hpp
// https://github.com/nvpro-samples/gl_render_vk_ddisplay
// https://github.com/KhronosGroup/Vulkan-Samples/blob/master/samples/extensions/open_gl_interop/open_gl_interop.cpp

#include "RenderBackendVKDirect.h"
#include "rhi/vk/RHIWindowRenderTargetVK.h"
#include "common/Timing.h"
#include "rhi/RHI.h"
#include <epoxy/egl.h>
#include <xf86drm.h>
#include <fcntl.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <dlfcn.h>
#include <atomic>
#include <cassert>
#include "rhi/gl/GLCommon.h"
#include <nvtx3/nvToolsExt.h>

#define CHECK(x)                                                                         \
  if (!(x)) {                                                                            \
    fprintf(stderr, "%s:%d: %s failed (%s)\n", __FILE__, __LINE__, #x, strerror(errno)); \
    abort();                                                                             \
  }
#define EGL_CHECK(x)                                                                   \
  if (!(x)) {                                                                          \
    fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, eglGetError()); \
    abort();                                                                           \
  }
// Vulkan dispatch loader storage lives in rhi/vk/RHIVulkan.cpp.

#ifdef IS_TEGRA
// Minimal reverse-engineered libnvrm_host1x syncpoint API.
//
// NvRmHost1xSyncpointWait exposes the kernel-recorded hardware timestamp of the syncpoint signal.
// In the absence of any of the advanced Vulkan presentation timing extensions,
// this provides better accuracy than the CPU-timestamped Vulkan display event codepath.
extern "C" {
struct NvRmHost1xTimestamp { // out-parameter struct populated by SyncpointWait
  uint32_t tv_sec;
  uint32_t _pad;
  uint32_t tv_nsec;
  uint32_t clock_id; // library-translated: kernel 0/1 -> 0, kernel 2 -> 1
};

uint32_t NvRmHost1xOpen(void** out_handle, uint32_t attrs);
void NvRmHost1xClose(void* handle);
uint32_t NvRmHost1xGetDefaultOpenAttrs(uint32_t* out_attrs);
uint32_t NvRmHost1xSyncpointRead(void* h1x, uint32_t id, uint32_t* out_value);
uint32_t NvRmHost1xWaiterAllocate(void** out_waiter, void* h1x);
void NvRmHost1xWaiterFree(void* waiter);
uint32_t NvRmHost1xSyncpointWait(void* waiter, uint32_t id, uint32_t thresh, uint32_t timeout_us, NvRmHost1xTimestamp* out_ts);
}

// VBlank syncpt, retrieved by hooking ioctl.
uint32_t vblankSyncptId = 0;

// ioctl shim for figuring out what syncpt channel our display vblank is on.
// Early in initialization, the display drivers call nvKmsIoctl with NVKMS_IOCTL_ENABLE_VBLANK_SYNC_OBJECT,
// which populates the syncpt ID in its response. This is the easiest way to find that.
// (The alternative is using the NvKms API ourselves correctly, and there's a lot going on there.)
static std::atomic_bool ioctl_shim_active = true;
static int (*real_ioctl)(int fd, unsigned long request, ...) = nullptr;

// Loader function for real_ioctl.
void init_ioctl_shim() { real_ioctl = reinterpret_cast<decltype(real_ioctl)>(dlsym(RTLD_NEXT, "ioctl")); }
// Load the real ioctl as early as possible by putting a function pointer into preinit_array.
__attribute__((section(".preinit_array"), used)) static void (*const preinit_ioctl_shim)(void) = init_ioctl_shim;


// NvKmsIoctl helper structs
struct NvKmsIoctlArg {
  uint32_t command;
  uint32_t payloadLength;
  void* payload;
};

struct NvKmsEnableVblankSyncObjectRequest {
  uint32_t deviceHandle;
  uint32_t dispHandle;
  uint32_t head;
};

struct NvKmsEnableVblankSyncObjectReply {
  uint32_t vblankHandle;
  uint32_t syncptId;
};

struct NvKmsEnableVblankSyncObjectParams {
  struct NvKmsEnableVblankSyncObjectRequest request; /*! in */
  struct NvKmsEnableVblankSyncObjectReply reply; /*! out */
};

constexpr unsigned long kNvKmsIoctlRequest = 0xc0106d00;
constexpr uint32_t NVKMS_IOCTL_ENABLE_VBLANK_SYNC_OBJECT = 56;

// Cold-path ioctl shim. This will self-disable once it catches the vblank sync object.
__attribute__((cold, noinline)) static int ioctl_shim(int fd, unsigned long request, void* arg) {
  NvKmsIoctlArg* kmsArg = (NvKmsIoctlArg*) arg;
  if (__builtin_expect((request == kNvKmsIoctlRequest && kmsArg->command == NVKMS_IOCTL_ENABLE_VBLANK_SYNC_OBJECT), false)) {
    int res = real_ioctl(fd, request, arg);
    if (res == 0) {
      // IOCTL succeeded, we should have the syncpt id.
      NvKmsEnableVblankSyncObjectParams* vblankParams = (NvKmsEnableVblankSyncObjectParams*) kmsArg->payload;
      fprintf(stderr, "NVKMS_IOCTL_ENABLE_VBLANK_SYNC_OBJECT(%u, %u, %u) => syncpt %u\n",
        vblankParams->request.deviceHandle, vblankParams->request.dispHandle, vblankParams->request.head,
        vblankParams->reply.syncptId);

      vblankSyncptId = vblankParams->reply.syncptId;

      // Stop running the shim, we're done.
      std::atomic_store_explicit(&ioctl_shim_active, false, std::memory_order_relaxed);
    }
    return res;
  }
  return real_ioctl(fd, request, arg);
}

// Hot-path ioctl replacement. This is optimized to forward to the real ioctl as fast as possible once the shim self-disables.
extern "C" int ioctl(int fd, unsigned long request, ...) {
  va_list ap;
  va_start(ap, request);
  void* arg = va_arg(ap, void*);
  va_end(ap);

  if (__builtin_expect(
        std::atomic_load_explicit(&ioctl_shim_active, std::memory_order_relaxed), 0)) {
    return ioctl_shim(fd, request, arg);
  }
  return real_ioctl(fd, request, arg);
}
#endif // IS_TEGRA

RenderBackend* createVKDirectBackend() { return new RenderBackendVKDirect(); }

template <typename T>
bool contains(const std::vector<T>& container, const T& value) {
  return std::find(container.begin(), container.end(), value) != container.end();
}

RenderBackendVKDirect::RenderBackendVKDirect() {
}

void RenderBackendVKDirect::createGLContext() {
  // EGL display setup. Always required on Tegra: Argus's only supported
  // buffer type is EGL_IMAGE, so ArgusCamera needs an EGLDisplay to give to
  // Argus's setEGLDisplay (the EGLImage is a lightweight handle over an
  // NvBuf; the storage is what VK + CUDA import via DMA-BUF FD). The
  // SURFACELESS_MESA platform never takes over the display hardware, so
  // it coexists with VK_KHR_display scanout.
  {
    // clang-format off
    EGLint attrs[] = {
      EGL_NONE
    };
    // clang-format on
    EGL_CHECK(m_eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, /*native_display=*/ nullptr, attrs));
    EGL_CHECK(eglInitialize(m_eglDisplay, NULL, NULL));
  }
}

void RenderBackendVKDirect::createPresentation() {
  try {
    if (!rhi()->vk()) {
      fprintf(stderr, "RenderBackendVKDirect::createPresentation: rhi()->vk() is null; cannot present\n");
      abort();
    }
    vk::Instance instance = rhi()->vk()->instance();
    vk::PhysicalDevice gpu = rhi()->vk()->physicalDevice();
    vk::Device device = rhi()->vk()->device();
    uint32_t queueFamily = rhi()->vk()->queueFamilyIndex();

    if (gpu.getDisplayPropertiesKHR().empty()) {
      fprintf(stderr, "VKDirect: selected physical device exposes no displays via vkGetDisplayPropertiesKHR\n");
      abort();
    }

    // Select display. TODO: currently using the first available display.
    {
      auto displays = gpu.getDisplayPropertiesKHR();
      CHECK(!displays.empty());

      m_display.m_displayProperties = displays[0];
      m_display.m_displayKHR = m_display.m_displayProperties.display;
    }

    // Physical device properties enumeration
#if 0 // Disabled for compatibility with L4T r32.2 -- extension isn't present, but we don't really need it anyway
  {
    auto res = gpu.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDrmPropertiesEXT>();
    vk::PhysicalDeviceProperties& pdp = res.get<vk::PhysicalDeviceProperties2>().properties;
    vk::PhysicalDeviceDrmPropertiesEXT& drmExt = res.get<vk::PhysicalDeviceDrmPropertiesEXT>();
    printf("Device name: %s\n", pdp.deviceName.data());
    printf("DRM info: \n");
    printf("  hasPrimary=%u primary=(%ld, %ld)\n", drmExt.hasPrimary, drmExt.primaryMajor, drmExt.primaryMinor);
    printf("  hasRender=%u   render=(%ld, %ld)\n", drmExt.hasRender, drmExt.renderMajor, drmExt.renderMinor);

  }
#endif

    // Select mode and target plane; create display surface.
    {
      auto modes = gpu.getDisplayModePropertiesKHR(m_display.m_displayKHR);
      m_display.m_modeProperties = modes[0];
      for (auto& m : modes) {
        auto i = m.parameters.visibleRegion;
        auto c = m_display.m_modeProperties.parameters.visibleRegion;
        // Select for highest refresh rate at highest resolution
        if (((i.height * i.width) > (c.height * c.width)) ||
          (((i.height * i.width) == (c.height * c.width)) && m.parameters.refreshRate > m_display.m_modeProperties.parameters.refreshRate)) {
          m_display.m_modeProperties = m;
        }
      }

      // Pick first compatible plane
      auto planes = gpu.getDisplayPlanePropertiesKHR();
      uint32_t planeIndex = 0;
      bool foundPlane = false;
      for (uint32_t i = 0; i < planes.size(); ++i) {
        auto p = planes[i];

        // Skip planes bound to different display
        if (p.currentDisplay && (p.currentDisplay != m_display.m_displayKHR))
          continue;

        auto supportedDisplays = gpu.getDisplayPlaneSupportedDisplaysKHR(i);

        for (auto& d : supportedDisplays) {
          if (d == m_display.m_displayKHR) {
            foundPlane = true;
            planeIndex = i;
            break;
          }
        }

        if (foundPlane)
          break;
      }
      CHECK(foundPlane);

      // Find alpha mode bit
      auto planeCapabilities = gpu.getDisplayPlaneCapabilitiesKHR(m_display.m_modeProperties.displayMode, planeIndex);
      vk::DisplayPlaneAlphaFlagBitsKHR alphaMode = vk::DisplayPlaneAlphaFlagBitsKHR::eOpaque;
      vk::DisplayPlaneAlphaFlagBitsKHR alphaModes[4] = {
        vk::DisplayPlaneAlphaFlagBitsKHR::eOpaque,
        vk::DisplayPlaneAlphaFlagBitsKHR::eGlobal,
        vk::DisplayPlaneAlphaFlagBitsKHR::ePerPixel,
        vk::DisplayPlaneAlphaFlagBitsKHR::ePerPixelPremultiplied,
      };
      for (uint32_t i = 0; i < (sizeof(alphaModes) / sizeof(alphaModes[0])); ++i) {
        if (planeCapabilities.supportedAlpha & alphaModes[i]) {
          alphaMode = alphaModes[i];
          break;
        }
      }

      vk::DisplaySurfaceCreateInfoKHR surfaceCreateInfo{vk::DisplaySurfaceCreateFlagBitsKHR(),
        m_display.m_modeProperties.displayMode,
        planeIndex,
        planes[planeIndex].currentStackIndex,
        vk::SurfaceTransformFlagBitsKHR::eIdentity,
        1.0f,
        alphaMode,
        vk::Extent2D(m_display.m_modeProperties.parameters.visibleRegion.width,
          m_display.m_modeProperties.parameters.visibleRegion.height)};

      m_surface = instance.createDisplayPlaneSurfaceKHRUnique(surfaceCreateInfo);

      const auto& d = m_display.m_displayProperties;
      printf("Using display: %s\n  physical resolution: %i x %i\n", d.displayName, d.physicalResolution.width, d.physicalResolution.height);
      const auto& m = m_display.m_modeProperties;
      m_refreshRateHz = static_cast<double>(m.parameters.refreshRate) / 1000.0;
      printf("Display mode: %i x %i @ %fHz\n", m.parameters.visibleRegion.width, m.parameters.visibleRegion.height, m_refreshRateHz);
    }

    // Verify the queue family selected by RHIVulkan supports presentation on
    // this surface. The shared queue was selected solely on graphics support;
    // on Tegra it also supports present, but check rather than assume.
    if (!gpu.getSurfaceSupportKHR(queueFamily, m_surface.get())) {
      fprintf(stderr, "VKDirect: shared queue family %u does not support presentation on the display surface\n", queueFamily);
      abort();
    }

    // Create swapchain
    {
      auto formats = gpu.getSurfaceFormatsKHR(m_surface.get());
      auto capabilities = gpu.getSurfaceCapabilitiesKHR(m_surface.get());
      auto presentModes = gpu.getSurfacePresentModesKHR(m_surface.get());

      printf("Supported swapchain formats:\n");
      for (size_t i = 0; i < formats.size(); ++i) {
        printf("  [%zu] %s %s\n", i, to_string(formats[i].format).c_str(), to_string(formats[i].colorSpace).c_str());
      }

      // image count depending on capabilities
      uint32_t imageCount = std::min(capabilities.maxImageCount, capabilities.minImageCount + 1);

      // Pick a format matching the interop textures (R8G8B8A8 / GL_RGBA8) so the
      // blit can use a straight copy. A8B8G8R8UnormPack32 is byte-identical to
      // R8G8B8A8Unorm on little-endian. Fall back to B8G8R8A8, then whatever the
      // display supports.
      vk::SurfaceFormatKHR preferredFormats[] = {
        {      vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear},
        {vk::Format::eA8B8G8R8UnormPack32, vk::ColorSpaceKHR::eSrgbNonlinear},
        {      vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear},
      };
      vk::SurfaceFormatKHR format = formats[0]; // default fallback
      if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
        format = preferredFormats[0];
      } else {
        for (auto& pf : preferredFormats) {
          if (contains(formats, pf)) {
            format = pf;
            break;
          }
        }
      }

      printf("Selected swapchain format: %s %s\n", to_string(format.format).c_str(), to_string(format.colorSpace).c_str());

      // use valid extent if available, otherwise derive from display mode
      vk::Extent2D extent;
      if (capabilities.currentExtent.width == 0xFFFFFFFF) {
        extent = m_display.m_modeProperties.parameters.visibleRegion;

        auto clamp = [](int val, int min, int max) { return (val < min) ? min : (val > max) ? max
                                                                                            : val; };
        extent.width = clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
      } else {
        extent = capabilities.currentExtent;
      }

      vk::SurfaceTransformFlagBitsKHR pretransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
      if ((pretransform & capabilities.supportedTransforms) != pretransform) {
        pretransform = capabilities.currentTransform;
      }

      printf("Supported presentation modes: ");
      for (auto& m : presentModes) {
        printf("%s ", to_string(m).c_str());
      }

      // Select a suitable presentation mode. eFifo is required to be supported so that'll be our fallback.
      vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
      if (contains(presentModes, vk::PresentModeKHR::eMailbox)) { // Mailbox: optimal
        presentMode = vk::PresentModeKHR::eMailbox;
      } else if (contains(presentModes, vk::PresentModeKHR::eImmediate)) { // Immediate might tear, but it'll keep latency low
        presentMode = vk::PresentModeKHR::eImmediate;
      }
      // RHI_VK_PRESENT_MODE env var overrides (e.g. for stress testing the
      // VK-native sync). Accepts "fifo", "mailbox", "immediate".
      if (const char* envMode = getenv("RHI_VK_PRESENT_MODE")) {
        if (strcmp(envMode, "fifo") == 0)
          presentMode = vk::PresentModeKHR::eFifo;
        else if (strcmp(envMode, "mailbox") == 0 && contains(presentModes, vk::PresentModeKHR::eMailbox))
          presentMode = vk::PresentModeKHR::eMailbox;
        else if (strcmp(envMode, "immediate") == 0 && contains(presentModes, vk::PresentModeKHR::eImmediate))
          presentMode = vk::PresentModeKHR::eImmediate;
      }

      printf("\nSelected presentation mode: %s\n", to_string(presentMode).c_str());

      // VK_KHR_display
      // create swapchain using the ddisplay surface created before

      vk::SwapchainCreateInfoKHR swapchainCreateInfo{vk::SwapchainCreateFlagsKHR(),
        m_surface.get(),
        imageCount,
        format.format,
        format.colorSpace,
        extent,
        1,
        vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst),
        vk::SharingMode::eExclusive,
        0,
        nullptr,
        pretransform,
        vk::CompositeAlphaFlagBitsKHR::eOpaque,
        presentMode,
        VK_TRUE};

      m_swapchain = device.createSwapchainKHRUnique(swapchainCreateInfo);
      m_swapchainImages = device.getSwapchainImagesKHR(m_swapchain.get());
      CHECK(!m_swapchainImages.empty());
      m_swapchainExtent = extent;
      m_swapchainFormat = format.format;
    }

    // Swapchain management semaphores
    // (acquireNextImageKHR signals imageAcquired)
    for (size_t i = 0; i < m_swapchainImages.size(); ++i) {
      vk::SemaphoreCreateInfo ci{};
      m_imageAcquiredSemaphores.push_back(device.createSemaphoreUnique(ci));
    }
    // VK-native: additional per-swap-image renderFinished semaphores so the
    // signal/wait sequencing on the presentation semaphore is safe across
    // reordered present modes. See member comment in the header.
    for (size_t i = 0; i < m_swapchainImages.size(); ++i) {
      vk::SemaphoreCreateInfo ci{};
      m_renderFinishedSemaphoresPerImage.push_back(device.createSemaphoreUnique(ci));
      // Acquire fences: created unsignaled; acquireVKFrame resets before use.
      m_imageAcquiredFences.push_back(device.createFenceUnique(vk::FenceCreateInfo{}));
    }

    // VK-native: lightweight handle that RHIVK updates with the
    // currently-acquired swap image per frame.
    m_windowRenderTarget = new RHIWindowRenderTargetVK(
      m_swapchainExtent.width, m_swapchainExtent.height, m_swapchainFormat);


    // Scanout timestamp source setup.

#ifdef IS_TEGRA
    // Tegra: host1x vblank syncpoint. If init fails, leave the worker thread
    // unstarted; lastPresentationTimestamp stays at 0. The render loop will
    // continue to function; only consumers that depend on the timestamp are
    // affected.

    // We first need to render a frame to the device to get the output configured;
    // without this, the ioctl hook can't catch the vblank syncpt ID.
    {
      VKFrameInfo frameInfo = acquireVKFrame();
      // We need to submit a command buffer that just clears this swapchain image.

      // Create a temporary command pool and a single command buffer.
      vk::CommandPoolCreateInfo commandPoolCreateInfo = {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily};
      vk::UniqueCommandPool commandPool = device.createCommandPoolUnique(commandPoolCreateInfo);

      vk::CommandBufferAllocateInfo commandBufferAllocateInfo = {commandPool.get(), vk::CommandBufferLevel::ePrimary, 1};
      auto commandBuffers = device.allocateCommandBuffersUnique(commandBufferAllocateInfo);

      auto cb = std::move(commandBuffers[0]);


      // Record a one-time-submit command buffer that clears the freshly
      // acquired swap image to black and leaves it ready for presentation.
      cb->begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

      vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

      // UNDEFINED -> TRANSFER_DST_OPTIMAL: discard prior contents, ready to clear.
      vk::ImageMemoryBarrier toTransfer{
        vk::AccessFlags(), vk::AccessFlagBits::eTransferWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        frameInfo.swapchainImage, range};
      cb->pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
        vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &toTransfer);

      vk::ClearColorValue clearColor{
        std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}
      };
      cb->clearColorImage(frameInfo.swapchainImage, vk::ImageLayout::eTransferDstOptimal, clearColor, range);

      // TRANSFER_DST_OPTIMAL -> PRESENT_SRC_KHR: hand the image to the presentation engine.
      vk::ImageMemoryBarrier toPresent{
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlags(),
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::ePresentSrcKHR,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        frameInfo.swapchainImage, range};
      cb->pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eBottomOfPipe,
        vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &toPresent);

      cb->end();

      // Submit, signaling renderFinished so the following present waits on the
      // clear. acquireVKFrame() host-waits on a fence and returns a null
      // imageAcquired semaphore, so there is normally no GPU-side acquire wait
      // to add; guard on it in case that contract ever changes.
      vk::CommandBuffer rawCb = cb.get();
      vk::Semaphore waitSem = frameInfo.imageAcquired;
      vk::Semaphore signalSem = frameInfo.renderFinished;
      vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eTransfer;

      vk::SubmitInfo submit;
      submit.setCommandBuffers(rawCb);
      submit.setSignalSemaphores(signalSem);
      if (waitSem) {
        submit.setWaitSemaphores(waitSem);
        submit.setWaitDstStageMask(waitStage);
      }
      rhi()->vk()->queue().submit(submit, vk::Fence());

      // Present (waits on renderFinished), then drain the queue -- including this
      // present -- before proceeding. This synchronous drain is the whole point
      // of the prologue: it forces the driver to configure its scanout path so
      // the following init code can snoop the vblank syncpt.
      presentVKFrame(frameInfo);
      rhi()->vk()->queue().waitIdle();

      // commandPool and cb are locals scoped to this block: the waitIdle above
      // guarantees the GPU is done with them, and RAII frees the buffer before
      // destroying its pool (pool declared first => destroyed last).
    }

    if (initScanoutSyncpt()) {
      m_scanoutThread = FxThread(&RenderBackendVKDirect::scanoutThreadFunc, this);
    } else {
      fprintf(stderr, "RenderBackendVKDirect: host1x syncpt init failed; lastPresentationTimestamp will remain 0\n");
    }
#else
    // Non-Tegra: Vulkan display-event fence path.
    m_scanoutEventFd = eventfd(0, 0);
    CHECK(m_scanoutEventFd >= 0);
    m_scanoutThread = FxThread(&RenderBackendVKDirect::scanoutThreadFunc, this);
#endif

  } catch (const std::exception& ex) {
    printf("%s\n", ex.what());
    abort();
  }
}

RenderBackendVKDirect::~RenderBackendVKDirect() {
#ifdef IS_TEGRA
  // Signal the worker to exit. It's blocking in NvRmHost1xSyncpointWait with
  // a 500ms timeout, which bounds shutdown latency.
  m_scanoutShutdown.store(true, std::memory_order_release);
#else
  if (m_scanoutEventFd >= 0) {
    // Closing the eventfd makes the worker's read() return 0 immediately.
    close(m_scanoutEventFd);
    m_scanoutEventFd = -1;
  }
#endif
  if (m_scanoutThread.joinable())
    m_scanoutThread.join();

  // Drain in-flight GPU work before destroying VK objects. Use the RHI's
  // bounded drain rather than device.waitIdle(): by this point presentation has
  // stopped, so the graphics queue may be parked forever on the swapchain
  // imageAcquired semaphore (see RHIVK::waitForGPUIdle). An unbounded waitIdle()
  // here would block until the GPU host watchdog fires and loses the device,
  // wedging the display driver -- the very crash this teardown is trying to
  // avoid. RHIVulkan owns the device; if its context is gone there's nothing to
  // wait on.
  if (rhi() && rhi()->vk()) {
    rhi()->waitForGPUIdle();

    // If the GPU is already lost, any further VK teardown is unsafe: on this
    // driver vkDestroySwapchainKHR (and friends) busy-spin forever instead of
    // returning. Nothing in-process can recover a lost device, so terminate
    // immediately rather than hang. Process teardown reclaims the resources.
    if (rhi()->isDeviceLost()) {
      fprintf(stderr, "RenderBackendVKDirect: device lost during shutdown; exiting to avoid a hang in VK teardown.\n");
      fflush(stderr);
      _exit(0);
    }

#ifndef IS_TEGRA
    // Destroy any fence left in the mailbox.
    vk::Device device = rhi()->vk()->device();
    VkFence leftover = m_scanoutFenceMailbox.load(std::memory_order_acquire);
    if (leftover != VK_NULL_HANDLE)
      device.destroyFence(leftover);
#endif
  }

#ifdef IS_TEGRA
  // Host1x shutdown
  if (m_nvrmHost1x) {
    NvRmHost1xClose(m_nvrmHost1x);
    m_nvrmHost1x = nullptr;
  }
#endif
}

#ifdef IS_TEGRA
bool RenderBackendVKDirect::initScanoutSyncpt() {
  if (vblankSyncptId == 0) {
    fprintf(stderr, "RenderBackendVKDirect::initScanoutSyncpt(): Syncpt ID is unknown!\n");
    return false;
  }

  uint32_t attrs = 0;
  // NvRmHost1xGetDefaultOpenAttrs is a complete no-op and returns 1 (failure) on the current build.
  // The Vulkan ICD init path calls it like this, though, so we'll do the same.
  NvRmHost1xGetDefaultOpenAttrs(&attrs);

  uint32_t rc = NvRmHost1xOpen(&m_nvrmHost1x, attrs);
  if (rc != 0 || !m_nvrmHost1x) {
    fprintf(stderr, "RenderBackendVKDirect::initScanoutSyncpt(): NvRmHost1xOpen failed rc=0x%x\n", rc);
    return false;
  }

  // Sanity-read of the target syncpoint. If this fails the id is likely wrong
  // (e.g., nvkms-fence moved to a different id after a kernel update); falling
  // back to the fence path is safer than running with a stale id that will
  // silently never tick.
  uint32_t value = 0;
  rc = NvRmHost1xSyncpointRead(m_nvrmHost1x, vblankSyncptId, &value);
  if (rc != 0) {
    fprintf(stderr, "RenderBackendVKDirect::initScanoutSyncpt(): NvRmHost1xSyncpointRead(id=%u) failed rc=0x%x\n", vblankSyncptId, rc);
    NvRmHost1xClose(m_nvrmHost1x);
    m_nvrmHost1x = nullptr;
    return false;
  }
  printf("RenderBackendVKDirect::initScanoutSyncpt(): using host1x syncpt id=%u (current value %u)\n", vblankSyncptId, value);

  return true;
}

void RenderBackendVKDirect::scanoutThreadFunc() {
  pthread_setname_np(pthread_self(), "VKDirectScanout");
  // The waiter is reusable, so we just allocate it once at thread startup.
  void* waiter = nullptr;
  if (NvRmHost1xWaiterAllocate(&waiter, m_nvrmHost1x) != 0 || !waiter) {
    fprintf(stderr, "RenderBackendVKDirect::scanoutThreadFunc(): NvRmHost1xWaiterAllocate failed!\n");
    return;
  }

  // Get the initial value to start the wait loop.
  uint32_t current = 0;
  NvRmHost1xSyncpointRead(m_nvrmHost1x, vblankSyncptId, &current);

  // Finite timeout so shutdown can be checked between waits even if the
  // display isn't ticking. 500ms is short enough for responsive shutdown
  // and long enough that a live display won't hit it.
  const uint32_t kWaitTimeoutUs = 500000;

  while (!m_scanoutShutdown.load(std::memory_order_acquire)) {
    NvRmHost1xTimestamp ts = {};
    uint32_t rc = NvRmHost1xSyncpointWait(waiter, vblankSyncptId, current + 1, kWaitTimeoutUs, &ts);
    nvtxMarkA("VBlank syncpoint wait finished");

    if (rc == 0) {
      // Translate kernel CLOCK_MONOTONIC ns into our CNTVCT-based timebase
      // so consumers reading lastPresentationTimestamp() can compare against
      // currentTimeNs() directly.
      uint64_t kernel_ns = (uint64_t) ts.tv_sec * 1000000000ULL + ts.tv_nsec;

      // Compute CLOCK_MONOTONIC to CNTVCT timebase offset.
      // Bracket the clock_gettime() call with two cntvct reads and use the midpoint to improve accuracy.
      // We have to continually calculate the offset because CLOCK_MONOTONIC is gradually adjusted by NTP sync.
      // (if it were CLOCK_MONOTONIC_RAW, that'd be a one-time fixed offset computation.)
      uint64_t monoToTscOffsetNs;
      {
        struct timespec mts;
        uint64_t cntvct_before, cntvct_after;
        asm volatile("mrs %0, cntvct_el0"
                     : "=r"(cntvct_before));
        clock_gettime(CLOCK_MONOTONIC, &mts);
        asm volatile("mrs %0, cntvct_el0"
                     : "=r"(cntvct_after));
        uint64_t cntvct_mid_ns = tscTimestampToNs((cntvct_before + cntvct_after) / 2);
        uint64_t mono_ns = (uint64_t) mts.tv_sec * 1000000000ULL + mts.tv_nsec;
        monoToTscOffsetNs = (int64_t) cntvct_mid_ns - (int64_t) mono_ns;
      }

      uint64_t tsc_ns = (uint64_t) ((int64_t) kernel_ns + monoToTscOffsetNs);
      m_lastPresentationTimestamp.store(tsc_ns, std::memory_order_release);
    }
    // Re-read the current value so the next threshold is always "next
    // unseen tick," even if we lost a vblank (e.g., to timeout or
    // scheduling). Protects against returning immediately on stale data.
    NvRmHost1xSyncpointRead(m_nvrmHost1x, vblankSyncptId, &current);
  }
  NvRmHost1xWaiterFree(waiter);
}

#else

// Original Vulkan display timing backend. More jitter due to timestamping on the CPU after thread wakeup.

// Margin before the expected scanout time at which we switch from
// blocking vkWaitForFences to WFE spin-polling. Trades a small amount of
// power for lower wakeup jitter — WFE power-gates the core between polls.
constexpr uint64_t kScanoutSpinMarginNs = 500000; // 500 µs

void RenderBackendVKDirect::scanoutThreadFunc() {
  pthread_setname_np(pthread_self(), "VKDirectScanout");
  vk::Device device = rhi()->vk()->device();
  const uint64_t refreshPeriodNs = static_cast<uint64_t>(1000000000.0 / m_refreshRateHz);
  bool useAdaptiveBlock = false;
  {
    const char* s = getenv("VKDIRECT_USE_ADAPTIVE_BLOCK");
    if (s) {
      useAdaptiveBlock = (s[0] == '1');
    }
  }
  if (useAdaptiveBlock) {
    printf("RenderBackendVKDirect::scanoutThreadFunc(): Using adaptive block strategy\n");
  }

  for (;;) {
    // Block until the main thread posts a new fence (or the eventfd is closed).
    uint64_t val;
    if (read(m_scanoutEventFd, &val, sizeof(val)) != sizeof(val))
      break; // eventfd closed — time to shut down

    // Pick up the fence from the mailbox
    VkFence rawFence = m_scanoutFenceMailbox.exchange(VK_NULL_HANDLE, std::memory_order_acq_rel);
    if (rawFence == VK_NULL_HANDLE)
      continue;

    vk::Fence fence(rawFence);

    if (useAdaptiveBlock) {
      // Adaptive block + WFE spin:
      // 1) Estimate when this scanout will fire based on the last one + refresh period.
      // 2) Block (kernel sleep) until we're within kScanoutSpinMarginNs of that estimate.
      // 3) WFE spin-poll for the final stretch — low power, low jitter.
      uint64_t lastTs = m_lastPresentationTimestamp.load(std::memory_order_acquire);
      if (lastTs != 0) {
        uint64_t expectedScanout = lastTs + refreshPeriodNs;
        uint64_t now = currentTimeNs();
        if (expectedScanout > now + kScanoutSpinMarginNs) {
          // Block for the bulk of the wait. Convert to Vulkan timeout (nanoseconds).
          uint64_t blockNs = expectedScanout - now - kScanoutSpinMarginNs;
          // Timeout is intentional — we spin-poll for the final stretch below.
          (void) device.waitForFences(fence, VK_TRUE, blockNs);
        }
      } else {
        // No prior timestamp — first frame. Use a coarse blocking wait that leaves
        // margin for the spin phase (one full refresh period minus the spin margin).
        uint64_t blockNs = refreshPeriodNs > kScanoutSpinMarginNs
          ? refreshPeriodNs - kScanoutSpinMarginNs
          : 0;
        if (blockNs > 0)
          // Timeout is intentional — we spin-poll for the final stretch below.
          (void) device.waitForFences(fence, VK_TRUE, blockNs);
      }

      // WFE spin-poll: power-gate the core between status checks.
      while (device.getFenceStatus(fence) == vk::Result::eNotReady) {
#ifdef __aarch64__
        asm volatile("wfe" ::
                       : "memory");
#else
        // x86 fallback — yield to the hyperthread / save a tiny bit of power
        asm volatile("pause" ::
                       : "memory");
#endif
      }
    } else {
      // Basic waitForFences strategy. Infinite timeout; result is always eSuccess.
      (void) device.waitForFences(fence, VK_TRUE, std::numeric_limits<uint64_t>::max());
    }

    m_lastPresentationTimestamp.store(currentTimeNs(), std::memory_order_release);
    nvtxMarkA("Scanout fence wait finished");

    device.destroyFence(fence);
  }
}
#endif // IS_TEGRA

// VKFrameSource implementation. Hands out the next swap image + per-frame
// sync primitives. RHIVK records render commands directly into the image,
// then submit + present via presentVKFrame.
VKFrameInfo RenderBackendVKDirect::acquireVKFrame() {
  vk::Device device = rhi()->vk()->device();

  // Acquire with a fence (not a semaphore) and wait host-side. This keeps
  // image-readiness entirely CPU-side, so the render submit that RHIVK builds
  // never has to wait on an imageAcquired semaphore. That matters at shutdown:
  // a semaphore-based acquire parks the graphics queue on the presentation
  // engine, and once presentation stops the image pool deadlocks (every image
  // acquired, none freed), the GPU host watchdog fires, and the device is lost.
  // With a host wait there is no GPU-side acquire to strand.
  vk::Fence acquireFence = m_imageAcquiredFences[m_frameIndex].get();
  device.resetFences(acquireFence);

  auto r = device.acquireNextImageKHR(m_swapchain.get(), std::numeric_limits<uint64_t>::max(), vk::Semaphore(), acquireFence);
  if (r.result == vk::Result::eSuboptimalKHR)
    fprintf(stderr, "RenderBackendVKDirect::acquireVKFrame: eSuboptimalKHR\n");
  uint32_t swapchainIndex = r.value;

  // Block until the presentation engine has released this image for rendering.
  (void) device.waitForFences(acquireFence, VK_TRUE, std::numeric_limits<uint64_t>::max());

  // renderFinished is indexed by swap image (not frame slot) — see member
  // comment. Safe because acquireNextImageKHR returning image I implies the
  // previous present that waited on renderFinished[I] has completed.
  vk::Semaphore renderFinished = m_renderFinishedSemaphoresPerImage[swapchainIndex].get();

  VKFrameInfo frame;
  frame.swapchainIndex = swapchainIndex;
  frame.swapchainImage = m_swapchainImages[swapchainIndex];
  frame.extent = m_swapchainExtent;
  frame.format = m_swapchainFormat;
  frame.imageAcquired = vk::Semaphore(); // host-synchronized; no GPU wait needed
  frame.renderFinished = renderFinished;
  return frame;
}

void RenderBackendVKDirect::presentVKFrame(const VKFrameInfo& frame) {
  vk::Queue presentQueue = rhi()->vk()->queue();

  // Present waits on renderFinished (which the caller's render submit was
  // required to signal). Store handles in named locals — taking & of
  // .get()'s rvalue would leave a dangling pointer in PresentInfoKHR after
  // the constructor expression ends.
  vk::SwapchainKHR swapchain = m_swapchain.get();
  vk::Semaphore waitSem = frame.renderFinished;
  uint32_t swapIdx = frame.swapchainIndex;
  vk::PresentInfoKHR presentInfo{1, &waitSem, 1, &swapchain, &swapIdx};
  vk::Result presentResult = presentQueue.presentKHR(presentInfo);
  if (presentResult == vk::Result::eSuboptimalKHR)
    fprintf(stderr, "RenderBackendVKDirect::presentVKFrame: eSuboptimalKHR\n");

  m_frameIndex = (m_frameIndex + 1) % m_swapchainImages.size();
}

uint32_t RenderBackendVKDirect::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
  vk::PhysicalDeviceMemoryProperties memProperties = rhi()->vk()->physicalDevice().getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find suitable memory type!");
}
