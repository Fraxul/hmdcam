// References:
// https://github.com/KhronosGroup/Vulkan-Hpp
// https://github.com/nvpro-samples/gl_render_vk_ddisplay
// https://github.com/KhronosGroup/Vulkan-Samples/blob/master/samples/extensions/open_gl_interop/open_gl_interop.cpp

#include "RenderBackendVKDirect.h"
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
  // EGL setup. Must run before initRHIVulkan(), since the Vulkan physical device is selected to UUID-match the GL context.
  // We use the SURFACELESS_MESA platform, since we explicitly do not want this EGL context to take over the display hardware.
  // (using the DEVICE_EXT platform with a DRM FD will prevent Vulkan from being able to create a swapchain)
  {
    // clang-format off
    EGLint attrs[] = {
      EGL_NONE
    };
    // clang-format on
    EGL_CHECK(m_eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, /*native_display=*/ nullptr, attrs));
    EGL_CHECK(eglInitialize(m_eglDisplay, NULL, NULL));

    EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    eglBindAPI(EGL_OPENGL_ES_API);

    EGL_CHECK(m_eglContext = eglCreateContext(m_eglDisplay, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr)); // EGL_KHR_no_config_context
    EGL_CHECK(eglMakeCurrent(m_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, m_eglContext));

    printf("OpenGL Renderer: %s || Version: %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
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

    // Create sync objects
    {
      m_syncData.resize(m_swapchainImages.size());
      for (auto& s : m_syncData) {
        // Interop texture

        // clang-format off
        vk::StructureChain<vk::ImageCreateInfo, vk::ExternalMemoryImageCreateInfo> imageCreateInfo = {
          vk::ImageCreateInfo({
            vk::ImageCreateFlags(),
             vk::ImageType::e2D,
             vk::Format::eR8G8B8A8Unorm,
             vk::Extent3D(m_swapchainExtent, 1),
             /*mipLevels=*/ 1,
             /*arrayLayers=*/ 1,
             vk::SampleCountFlagBits::e1,
             vk::ImageTiling::eOptimal,
             vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc),
             vk::SharingMode::eExclusive,
             /*queueFamilies=*/ 0, nullptr,
             /*initialLayout=*/ vk::ImageLayout::eUndefined
          }),
          vk::ExternalMemoryImageCreateInfo({
            vk::ExternalMemoryHandleTypeFlags(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd)
          }),
        };
        // clang-format on

        s.m_image = device.createImageUnique(imageCreateInfo.get());

        vk::MemoryRequirements memoryRequirements = device.getImageMemoryRequirements(s.m_image.get());
        vk::MemoryAllocateInfo memoryAllocateInfo{memoryRequirements.size, findMemoryType(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlags())};

        // pass in hint that we want to export this memory
        vk::ExportMemoryAllocateInfo exportMemoryAllocateInfo(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
        memoryAllocateInfo.setPNext(&exportMemoryAllocateInfo);

        s.m_deviceMemory = device.allocateMemoryUnique(memoryAllocateInfo);

        device.bindImageMemory(s.m_image.get(), s.m_deviceMemory.get(), 0);

        // create OpenGL interop data
        vk::MemoryGetFdInfoKHR getHandleInfo = {s.m_deviceMemory.get(), vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd};
        s.m_handle = device.getMemoryFdKHR(getHandleInfo);


        GL(glCreateMemoryObjectsEXT(1, &s.m_memoryObject));
        GL(glImportMemoryFdEXT(s.m_memoryObject, memoryRequirements.size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, s.m_handle));

        glGenTextures(1, &s.m_textureGL);
        glBindTexture(GL_TEXTURE_2D, s.m_textureGL);
        GL(glTexStorageMem2DEXT(GL_TEXTURE_2D, /*levels=*/ 1, GL_RGBA8, m_swapchainExtent.width, m_swapchainExtent.height, s.m_memoryObject, /*offset=*/ 0));

        GL(glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, (GLint*) &s.m_internalFormat));

        GL(glGenFramebuffers(1, &s.m_framebufferGL));
        GL(glBindFramebuffer(GL_FRAMEBUFFER, s.m_framebufferGL));
        GL(glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, s.m_textureGL, /*level=*/ 0));

        GLenum framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        CHECK(framebufferStatus == GL_FRAMEBUFFER_COMPLETE);


        // Interop Semaphore
        vk::SemaphoreCreateInfo createInfo{};
        vk::ExportSemaphoreCreateInfo exportCreateInfo{vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd};
        createInfo.setPNext(&exportCreateInfo);

        auto makeSemaphore = [&](vk::UniqueSemaphore& s, int& fd, GLuint& g) {
          s = device.createSemaphoreUnique(createInfo);
          vk::SemaphoreGetFdInfoKHR getHandleInfo = {s.get(), vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd};
          fd = device.getSemaphoreFdKHR(getHandleInfo);

          glGenSemaphoresEXT(1, &g);
          GL(glImportSemaphoreFdEXT(g, GL_HANDLE_TYPE_OPAQUE_FD_EXT, fd));
        };

        makeSemaphore(s.m_available, s.m_availableHandle, s.m_availableGL);
        makeSemaphore(s.m_finished, s.m_finishedHandle, s.m_finishedGL);
      }
    }

    // Swapchain management semaphores
    for (size_t i = 0; i < m_swapchainImages.size(); ++i) {
      vk::SemaphoreCreateInfo ci{};
      m_imageAcquiredSemaphores.push_back(device.createSemaphoreUnique(ci));
      m_blitFinishedSemaphores.push_back(device.createSemaphoreUnique(ci));
    }

    // Command pool and command buffers (recorded per-frame in submitTexture)
    {
      vk::CommandPoolCreateInfo commandPoolCreateInfo = {vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily};
      m_commandPool = device.createCommandPoolUnique(commandPoolCreateInfo);

      vk::CommandBufferAllocateInfo commandBufferAllocateInfo = {m_commandPool.get(), vk::CommandBufferLevel::ePrimary, uint32_t(m_swapchainImages.size())};
      m_blitCommandBuffers = device.allocateCommandBuffers(commandBufferAllocateInfo);
    }

    m_unsignaledFrames = static_cast<uint32_t>(m_swapchainImages.size());

    // Window RT
    m_windowRenderTarget = new VKDirectSwapchainRenderTarget(this);
    m_windowRenderTarget->platformSetUpdatedWindowDimensions(surfaceWidth(), surfaceHeight());


    // Scanout timestamp source setup.
#ifdef IS_TEGRA
    // Tegra: host1x vblank syncpoint. If init fails, leave the worker thread
    // unstarted; lastPresentationTimestamp stays at 0. The render loop will
    // continue to function; only consumers that depend on the timestamp are
    // affected.

    // We first need to render a frame to the device to get the output configured;
    // without this, the ioctl hook can't catch the vblank syncpt ID.
    {
      VKGLSyncData* frame = acquireTexture();
      glBindFramebuffer(GL_FRAMEBUFFER, frame->m_framebufferGL);
      glViewport(0, 0, surfaceWidth(), surfaceHeight());
      glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
      glClear(GL_COLOR_BUFFER_BIT);
      glFlush();
      submitTexture(frame);
    }

    if (initScanoutSyncpt()) {
      m_scanoutThread = std::thread(&RenderBackendVKDirect::scanoutThreadFunc, this);
    } else {
      fprintf(stderr, "RenderBackendVKDirect: host1x syncpt init failed; lastPresentationTimestamp will remain 0\n");
    }
#else
    // Non-Tegra: Vulkan display-event fence path.
    m_scanoutEventFd = eventfd(0, 0);
    CHECK(m_scanoutEventFd >= 0);
    m_scanoutThread = std::thread(&RenderBackendVKDirect::scanoutThreadFunc, this);
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

  // Wait for all in-flight GPU work to complete before destroying VK objects.
  // RHIVulkan owns the device; if its allocator context is gone, there's
  // nothing to wait on.
  if (rhi() && rhi()->vk()) {
    vk::Device device = rhi()->vk()->device();
    device.waitIdle();

#ifndef IS_TEGRA
    // Destroy any fence left in the mailbox.
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

VKGLSyncData* RenderBackendVKDirect::acquireTexture() {
  if (m_unsignaledFrames > 0) {
    // Skip the wait for the first cycle through all swapchain images -- the
    // available semaphore for each index is first signaled by its blit submit,
    // so there is nobody to signal it on the very first use.
    --m_unsignaledFrames;
  } else {
    GLenum srcLayout = GL_LAYOUT_COLOR_ATTACHMENT_EXT;
    glWaitSemaphoreEXT(m_syncData[m_frameIndex].m_availableGL, 0, nullptr, 1, &m_syncData[m_frameIndex].m_textureGL, &srcLayout);
  }
  return &(m_syncData[m_frameIndex]);
}

void RenderBackendVKDirect::submitTexture(VKGLSyncData*) {
  vk::Device device = rhi()->vk()->device();
  vk::Queue presentQueue = rhi()->vk()->queue();

  // Synchronization overview (all semaphores are VK/GL-shared via EXT_external_semaphore):
  //
  //   acquireTexture:  GL waits on  "available"  -- blocks until VK is done reading the interop texture
  //                    ... GL renders into the interop texture ...
  //   submitTexture:   GL signals   "finished"   -- tells VK that GL rendering is complete
  //                    VK waits on  "finished" + "imageAcquired"
  //                    VK blits interop texture -> swapchain image
  //                    VK signals   "blitFinished" + "available"
  //                       blitFinished: gates presentKHR
  //                       available:    unblocks the next acquireTexture for this index

  // Signal VK that GL is done rendering into the interop texture.
  // The layout transition to TRANSFER_SRC prepares it for the VK blit read.
  GLenum targetLayout = GL_LAYOUT_TRANSFER_SRC_EXT;
  glSignalSemaphoreEXT(m_syncData[m_frameIndex].m_finishedGL, 0, nullptr, 1, &m_syncData[m_frameIndex].m_textureGL, &targetLayout);

  // Flush GL command stream so the semaphore signal is submitted to the GPU.
  glFlush();

  auto r = device.acquireNextImageKHR(m_swapchain.get(), std::numeric_limits<uint64_t>::max(), m_imageAcquiredSemaphores[m_frameIndex].get(), vk::Fence());
  if (r.result == vk::Result::eSuboptimalKHR)
    fprintf(stderr, "RenderBackendVKDirect: acquireNextImageKHR returned eSuboptimalKHR\n");
  uint32_t swapchainIndex = r.value;

  // Record the blit command buffer for this frame. The interop texture index
  // (m_frameIndex) and the swapchain image index (from acquireNextImageKHR) are
  // independent, so we record on the fly rather than pre-baking a 1:1 mapping.
  {
    auto& b = m_blitCommandBuffers[m_frameIndex];
    b.reset();
    b.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

    vk::ImageSubresourceRange subresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    // clang-format off
    vk::ImageMemoryBarrier swapchainTransferDstBarrier = {
      /*srcAccessMask=*/ vk::AccessFlagBits::eNone,
      /*dstAccessMask=*/ vk::AccessFlagBits::eTransferWrite,
      /*oldLayout=*/ vk::ImageLayout::eUndefined,
      /*newLayout=*/ vk::ImageLayout::eTransferDstOptimal,
      /*srcQueueFamilyIndex=*/ 0,
      /*dstQueueFamilyIndex=*/ 0,
      /*image=*/ m_swapchainImages[swapchainIndex],
      /*subresourceRange=*/ subresourceRange
    };
    // clang-format on

    b.pipelineBarrier(
      /*srcStageMask=*/ vk::PipelineStageFlagBits::eTopOfPipe,
      /*dstStageMask=*/ vk::PipelineStageFlagBits::eTransfer,
      /*dependencyFlags=*/ vk::DependencyFlagBits::eByRegion,
      /*memory barriers=*/ nullptr,
      /*buffer memory barriers=*/ nullptr,
      /*image memory barriers=*/ swapchainTransferDstBarrier);

    std::array<vk::Offset3D, 2> srcoffsets{
      vk::Offset3D{                               0,                                 0, 0},
      vk::Offset3D{int32_t(m_swapchainExtent.width), int32_t(m_swapchainExtent.height), 1}
    };
    std::array<vk::Offset3D, 2> dstoffsets{
      vk::Offset3D{                               0, int32_t(m_swapchainExtent.height), 0},
      vk::Offset3D{int32_t(m_swapchainExtent.width),                                 0, 1}
    };
    vk::ImageSubresourceLayers layers{vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    vk::ImageBlit region{layers, srcoffsets, layers, dstoffsets};

    b.blitImage(m_syncData[m_frameIndex].m_image.get(), vk::ImageLayout::eTransferSrcOptimal,
      m_swapchainImages[swapchainIndex], vk::ImageLayout::eTransferDstOptimal,
      vk::ArrayProxy<const vk::ImageBlit>{1, &region}, vk::Filter::eNearest);

    // clang-format off
    vk::ImageMemoryBarrier swapchainPresentSrcBarrier = {
      /*srcAccessMask=*/ vk::AccessFlagBits::eTransferWrite,
      /*dstAccessMask=*/ vk::AccessFlagBits::eNone,
      /*oldLayout=*/ vk::ImageLayout::eTransferDstOptimal,
      /*newLayout=*/ vk::ImageLayout::ePresentSrcKHR,
      /*srcQueueFamilyIndex=*/ 0,
      /*dstQueueFamilyIndex=*/ 0,
      /*image=*/ m_swapchainImages[swapchainIndex],
      /*subresourceRange=*/ subresourceRange
    };
    // clang-format on

    b.pipelineBarrier(
      /*srcStageMask=*/ vk::PipelineStageFlagBits::eTransfer,
      /*dstStageMask=*/ vk::PipelineStageFlagBits::eBottomOfPipe,
      /*dependencyFlags=*/ vk::DependencyFlagBits::eByRegion,
      /*memory barriers=*/ nullptr,
      /*buffer memory barriers=*/ nullptr,
      /*image memory barriers=*/ swapchainPresentSrcBarrier);

    b.end();
  }

  // Submit the blit.
  // Wait: "finished" (GL done) + "imageAcquired" (swapchain image ready).
  // Signal: "blitFinished" (for present) + "available" (GL can reuse this interop texture).
  std::array<vk::Semaphore, 2> blitWaitSemaphores{m_syncData[m_frameIndex].m_finished.get(), m_imageAcquiredSemaphores[m_frameIndex].get()};
  std::array<vk::PipelineStageFlags, 2> blitWaitStages{vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe};

  std::array<vk::Semaphore, 2> blitSignalSemaphores{m_blitFinishedSemaphores[m_frameIndex].get(), m_syncData[m_frameIndex].m_available.get()};

  vk::SubmitInfo submitInfo{blitWaitSemaphores,
    blitWaitStages,
    m_blitCommandBuffers[m_frameIndex],
    blitSignalSemaphores};

  presentQueue.submit(submitInfo, vk::Fence{});

  // Present: waits for the blit to finish before scanning out.
  std::vector<vk::Semaphore> presentWaitSemaphores{m_blitFinishedSemaphores[m_frameIndex].get()};
  vk::Result presentResult = presentQueue.presentKHR({presentWaitSemaphores, m_swapchain.get(), swapchainIndex});
  if (presentResult == vk::Result::eSuboptimalKHR)
    fprintf(stderr, "RenderBackendVKDirect: presentKHR returned eSuboptimalKHR\n");

#ifndef IS_TEGRA
  // VK_EXT_display_control: register a first-pixel-out fence and post it to
  // the scanout worker thread via atomic mailbox. Only compiled on non-Tegra
  // builds; on Tegra the worker thread waits directly on the host1x vblank
  // syncpoint and doesn't need per-frame fence registration.
  {
    vk::DisplayEventInfoEXT eventInfo{vk::DisplayEventTypeEXT::eFirstPixelOut};
    vk::Fence scanoutFence = device.registerDisplayEventEXT(m_display.m_displayKHR, eventInfo);

    // Exchange into the mailbox; destroy any stale fence we displaced.
    VkFence prev = m_scanoutFenceMailbox.exchange(static_cast<VkFence>(scanoutFence), std::memory_order_acq_rel);
    if (prev != VK_NULL_HANDLE)
      device.destroyFence(prev);

    // Wake the worker thread
    uint64_t val = 1;
    write(m_scanoutEventFd, &val, sizeof(val));
  }
#endif

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

VKDirectSwapchainRenderTarget::VKDirectSwapchainRenderTarget(RenderBackendVKDirect* backend) :
  m_backend(backend) {

  m_width = m_backend->surfaceWidth();
  m_height = m_backend->surfaceHeight();
  m_layers = 1;
  m_samples = 1;
  m_colorTargetCount = 1;
  m_isArray = false;
  m_hasDepthStencilTarget = false;
  internalAcquireTexture();
}

VKDirectSwapchainRenderTarget::~VKDirectSwapchainRenderTarget() {
  if (m_syncTex)
    m_backend->submitTexture(m_syncTex);
  m_syncTex = NULL;
}

void VKDirectSwapchainRenderTarget::platformSwapBuffers() {
  if (m_syncTex)
    m_backend->submitTexture(m_syncTex);

  internalAcquireTexture();
}
void VKDirectSwapchainRenderTarget::internalAcquireTexture() {
  m_syncTex = m_backend->acquireTexture();
  m_glFramebufferId = m_syncTex->m_framebufferGL;
}
