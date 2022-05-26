// References:
// https://github.com/KhronosGroup/Vulkan-Hpp
// https://github.com/nvpro-samples/gl_render_vk_ddisplay
// https://github.com/KhronosGroup/Vulkan-Samples/blob/master/samples/extensions/open_gl_interop/open_gl_interop.cpp

#include "RenderBackendVKDirect.h"
#include <epoxy/egl.h>
#include <xf86drm.h>
#include <fcntl.h>
#include "rhi/gl/GLCommon.h"

#define CHECK(x) if (!(x)) { fprintf(stderr, "%s:%d: %s failed (%s)\n", __FILE__, __LINE__, #x, strerror(errno)); abort(); }
#define EGL_CHECK(x) if (!(x)) { fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, eglGetError()); abort(); }

// Instantiate the vulkan dynamic loader in this file
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

RenderBackend* createVKDirectBackend() { return new RenderBackendVKDirect(); }

RenderBackendVKDirect::RenderBackendVKDirect() {

}

const std::vector<const char*> requiredInstanceExtensions = {
  VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
  VK_KHR_SURFACE_EXTENSION_NAME,
  VK_KHR_DISPLAY_EXTENSION_NAME,
  VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
  VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
};

const std::vector<const char*> requiredDeviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME,
  VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
  VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
  VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
  VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME
};

const std::vector<const char*> validationLayers = {
  "VK_LAYER_LUNARG_standard_validation"
};

void RenderBackendVKDirect::init() {
  try {
  // Load library and create instance
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = m_dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
  VULKAN_HPP_DEFAULT_DISPATCHER.init(vkGetInstanceProcAddr);

  bool enableValidation = false;
  {
    const char* s = getenv("VKDIRECT_ENABLE_VALIDATION");
    if (s) {
      enableValidation = (atoi(s) != 0);
    }
  }

  vk::InstanceCreateInfo createInfo{
      vk::InstanceCreateFlags(),
      /*applicationInfo=*/ nullptr,
      /*enabledLayers=*/ 0, nullptr,
      uint32_t(requiredInstanceExtensions.size()), requiredInstanceExtensions.data()
  };

  if (enableValidation) {
    createInfo.enabledLayerCount = (uint32_t) validationLayers.size();
    createInfo.ppEnabledLayerNames = validationLayers.data();
  }

  m_instance = vk::createInstanceUnique(createInfo);
  VULKAN_HPP_DEFAULT_DISPATCHER.init(m_instance.get());

  // Select GPU
  std::vector<vk::PhysicalDevice> devices = m_instance->enumeratePhysicalDevices();
  printf("%zu devices\n", devices.size());
  for(const auto& device : devices) {
    if(!device.getDisplayPropertiesKHR().empty()) {
      m_gpu = device;
      break;
    }
  }

  if (!m_gpu) {
    fprintf(stderr, "No GPU was able to provide display devices via vkGetDisplayPropertiesKHR\n");
    abort();
  }

  // Select display. TODO: currently using the first available display.
  {
    auto displays = m_gpu.getDisplayPropertiesKHR();
    CHECK(!displays.empty());

    m_display.m_displayProperties = displays[0];
    m_display.m_displayKHR = m_display.m_displayProperties.display;
  }

  // Physical device properties enumeration
  {
    auto res = m_gpu.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceDrmPropertiesEXT>();
    vk::PhysicalDeviceProperties& pdp = res.get<vk::PhysicalDeviceProperties2>().properties;
    vk::PhysicalDeviceDrmPropertiesEXT& drmExt = res.get<vk::PhysicalDeviceDrmPropertiesEXT>();
    printf("Device name: %s\n", pdp.deviceName.data());
    printf("DRM info: \n");
    printf("  hasPrimary=%u primary=(%ld, %ld)\n", drmExt.hasPrimary, drmExt.primaryMajor, drmExt.primaryMinor);
    printf("  hasRender=%u   render=(%ld, %ld)\n", drmExt.hasRender, drmExt.renderMajor, drmExt.renderMinor);

  }

  // Select mode and target plane; create display surface.
  {
    auto modes = m_gpu.getDisplayModePropertiesKHR(m_display.m_displayKHR);
    m_display.m_modeProperties = modes[0];
    for(auto& m : modes) {
      auto i = m.parameters.visibleRegion;
      auto c = m_display.m_modeProperties.parameters.visibleRegion;
      // Select for highest refresh rate at highest resolution
      if (((i.height * i.width) > (c.height * c.width)) ||
         (((i.height * i.width) == (c.height * c.width)) && m.parameters.refreshRate > m_display.m_modeProperties.parameters.refreshRate)) {
        m_display.m_modeProperties = m;
      }
    }

    // Pick first compatible plane
    auto planes = m_gpu.getDisplayPlanePropertiesKHR();
    uint32_t planeIndex = 0;
    bool foundPlane = false;
    for(uint32_t i = 0; i < planes.size(); ++i) {
      auto p = planes[i];

      // Skip planes bound to different display
      if(p.currentDisplay && (p.currentDisplay != m_display.m_displayKHR))
        continue;

      auto supportedDisplays = m_gpu.getDisplayPlaneSupportedDisplaysKHR(i);

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
    auto planeCapabilities = m_gpu.getDisplayPlaneCapabilitiesKHR(m_display.m_modeProperties.displayMode, planeIndex);
    vk::DisplayPlaneAlphaFlagBitsKHR alphaMode     = vk::DisplayPlaneAlphaFlagBitsKHR::eOpaque;
    vk::DisplayPlaneAlphaFlagBitsKHR alphaModes[4] = {vk::DisplayPlaneAlphaFlagBitsKHR::eOpaque,
                                                      vk::DisplayPlaneAlphaFlagBitsKHR::eGlobal,
                                                      vk::DisplayPlaneAlphaFlagBitsKHR::ePerPixel,
                                                      vk::DisplayPlaneAlphaFlagBitsKHR::ePerPixelPremultiplied};
    for(uint32_t i = 0; i < sizeof(alphaModes); i++)  {
      if(planeCapabilities.supportedAlpha & alphaModes[i]) {
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

    m_surface = m_instance->createDisplayPlaneSurfaceKHRUnique(surfaceCreateInfo);

    const auto& d = m_display.m_displayProperties;
    printf("Using display: %s\n  physical resolution: %i x %i\n", d.displayName, d.physicalResolution.width, d.physicalResolution.height);
    const auto& m = m_display.m_modeProperties;
    m_refreshRateHz = static_cast<double>(m.parameters.refreshRate) / 1000.0;
    printf("Display mode: %i x %i @ %fHz\n", m.parameters.visibleRegion.width, m.parameters.visibleRegion.height, m_refreshRateHz);
  }

  // Create logical device
  {
    // find graphics and present queue(s)
    auto families = m_gpu.getQueueFamilyProperties();
    bool found = false;
    for (uint32_t i = 0; i < families.size(); ++i) {
      if ((families[i].queueFlags & vk::QueueFlagBits::eGraphics) && (m_gpu.getSurfaceSupportKHR(i, m_surface.get()))) {
        // RFE: implement support for different (graphics != present) families
        m_presentFamily = i;
        found           = true;
      }
    }
    CHECK(found && "Found graphics and presentation queues");

    float priority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo{vk::DeviceQueueCreateFlags(), m_presentFamily, 1, &priority};
    vk::PhysicalDeviceFeatures deviceFeatures;

    // create the logical device and the present queue
    vk::DeviceCreateInfo deviceCreateInfo{vk::DeviceCreateFlags(),
                                          1,
                                          &queueCreateInfo,
                                          0,
                                          nullptr,
                                          uint32_t(requiredDeviceExtensions.size()),
                                          requiredDeviceExtensions.data(),
                                          &deviceFeatures};

    m_device = m_gpu.createDeviceUnique(deviceCreateInfo);
    m_presentQueue = m_device->getQueue(m_presentFamily, 0);

    VULKAN_HPP_DEFAULT_DISPATCHER.init(m_device.get());
  }

  // Create swapchain
  {
    auto formats      = m_gpu.getSurfaceFormatsKHR(m_surface.get());
    auto capabilities = m_gpu.getSurfaceCapabilitiesKHR(m_surface.get());
    auto presentModes = m_gpu.getSurfacePresentModesKHR(m_surface.get());

    // image count depending on capabilities
    uint32_t imageCount = std::min(capabilities.maxImageCount, capabilities.minImageCount + 1);

    // pick a preferred format or use the first available one
    vk::SurfaceFormatKHR format{vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
    bool valid = false;
    if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined) {
      valid = true;
    }
    for (auto& f : formats) {
      if (f == format) {
        valid = true;
        break;
      }
    }
    if (!valid) {
      format = formats[0];
    }

    // use valid extent if available, otherwise derive from display mode
    vk::Extent2D extent;
    if (capabilities.currentExtent.width == 0xFFFFFFFF) {
      extent = m_display.m_modeProperties.parameters.visibleRegion;

      auto clamp = [](int val, int min, int max) { return (val < min) ? min : (val > max) ? max : val; };
      extent.width  = clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
      extent.height = clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    } else {
      extent = capabilities.currentExtent;
    }

    vk::SurfaceTransformFlagBitsKHR pretransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
    if((pretransform & capabilities.supportedTransforms) != pretransform) {
      pretransform = capabilities.currentTransform;
    }

    // pick a preferred present mode or use fallback
    vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
    for(auto& m : presentModes) {
      if(m == vk::PresentModeKHR::eMailbox) {
        presentMode = m;
      }
    }

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

    m_swapchain = m_device->createSwapchainKHRUnique(swapchainCreateInfo);
    m_swapchainImages = m_device->getSwapchainImagesKHR(m_swapchain.get());
    CHECK(!m_swapchainImages.empty());
    m_swapchainExtent = extent;
    m_swapchainFormat = format.format;
  }

  // Set up EGL context
  {
    EGLint deviceCount = 0;
    EGL_CHECK(eglQueryDevicesEXT(0, NULL, &deviceCount));
    CHECK(deviceCount > 0);

    EGLDeviceEXT* eglDevices = new EGLDeviceEXT[deviceCount];
    EGL_CHECK(eglQueryDevicesEXT(deviceCount, eglDevices, &deviceCount));

    for (int i = 0; i < deviceCount; ++i) {
      const char* drmName = eglQueryDeviceStringEXT(eglDevices[i], EGL_DRM_DEVICE_FILE_EXT);
      fprintf(stderr, "EGL device [%d]: DRM file %s\n", i, drmName);
      if (!drmName)
        continue;

      if (!strcmp(drmName, "drm-nvdc")) {
        m_drmFd = drmOpen(drmName, NULL);
      } else {
        m_drmFd = open(drmName, O_RDWR, 0);
      }
      if (m_drmFd <= 0) {
        fprintf(stderr, "Unable to open DRM devices %s\n", drmName);
        continue;
      }

      m_eglDevice = eglDevices[i];
      fprintf(stderr, " -- Opened DRM device for EGL device %d\n", i);
      break;
    }

    delete[] eglDevices;
    if (!m_eglDevice) {
      fprintf(stderr, "Unable to open any DRM device.\n");
      abort();
    }

    EGLint attrs[] = {
      EGL_DRM_MASTER_FD_EXT, m_drmFd,
      EGL_NONE
    };
    EGL_CHECK(m_eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, m_eglDevice, attrs));
    EGL_CHECK(eglInitialize(m_eglDisplay, NULL, NULL));

    EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
    eglBindAPI(EGL_OPENGL_ES_API);

    EGL_CHECK(m_eglContext = eglCreateContext(m_eglDisplay, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr)); // EGL_KHR_no_config_context
    EGL_CHECK(eglMakeCurrent(m_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, m_eglContext));

    printf("OpenGL Renderer: %s || Version: %s\n", glGetString(GL_RENDERER), glGetString(GL_VERSION));
  }

  // Create sync objects
  {
    m_syncData.resize(m_swapchainImages.size());
    for(auto& s : m_syncData) {
      // Interop texture

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
           /*initialLayout=*/ vk::ImageLayout::eTransferSrcOptimal
        }),
        vk::ExternalMemoryImageCreateInfo({
          vk::ExternalMemoryHandleTypeFlags(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd)
        })};

      s.m_image = m_device->createImageUnique(imageCreateInfo.get());

      vk::MemoryRequirements memoryRequirements = m_device->getImageMemoryRequirements(s.m_image.get());
      vk::MemoryAllocateInfo memoryAllocateInfo{memoryRequirements.size, findMemoryType(memoryRequirements.memoryTypeBits, vk::MemoryPropertyFlags())};

      // pass in hint that we want to export this memory
      vk::ExportMemoryAllocateInfo exportMemoryAllocateInfo(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
      memoryAllocateInfo.setPNext(&exportMemoryAllocateInfo);

      s.m_deviceMemory = m_device->allocateMemoryUnique(memoryAllocateInfo);

      m_device->bindImageMemory(s.m_image.get(), s.m_deviceMemory.get(), 0);

      // create OpenGL interop data
      vk::MemoryGetFdInfoKHR getHandleInfo = { s.m_deviceMemory.get(), vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd };
      s.m_handle =  m_device->getMemoryFdKHR(getHandleInfo);


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
      vk::SemaphoreCreateInfo       createInfo{};
      vk::ExportSemaphoreCreateInfo exportCreateInfo{vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd};
      createInfo.setPNext(&exportCreateInfo);

      auto makeSemaphore = [&](vk::UniqueSemaphore& s, int& fd, GLuint& g) {
        s = m_device->createSemaphoreUnique(createInfo);
        vk::SemaphoreGetFdInfoKHR getHandleInfo = {s.get(), vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd };
        fd = m_device->getSemaphoreFdKHR(getHandleInfo);

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
    m_imageAcquiredSemaphores.push_back(m_device->createSemaphoreUnique(ci));
    m_blitFinishedSemaphores.push_back(m_device->createSemaphoreUnique(ci));
  }

  // Command pool and command buffers
  {
    vk::CommandPoolCreateInfo commandPoolCreateInfo = {vk::CommandPoolCreateFlags(), m_presentFamily};
    m_commandPool = m_device->createCommandPool(commandPoolCreateInfo);

    vk::CommandBufferAllocateInfo commandBufferAllocateInfo = {m_commandPool, vk::CommandBufferLevel::ePrimary, uint32_t(m_swapchainImages.size())};

    m_blitCommandBuffers = m_device->allocateCommandBuffers(commandBufferAllocateInfo);

    for (size_t i = 0; i < m_swapchainImages.size(); ++i) {
      auto& b = m_blitCommandBuffers[i];
      b.begin({ vk::CommandBufferUsageFlagBits::eSimultaneousUse });

      std::array<vk::Offset3D, 2> srcoffsets{ vk::Offset3D{ 0,0,0 }, vk::Offset3D{ int32_t(m_swapchainExtent.width), int32_t(m_swapchainExtent.height), 1 } };
      std::array<vk::Offset3D, 2> dstoffsets{ vk::Offset3D{ 0,int32_t(m_swapchainExtent.height),0 }, vk::Offset3D{ int32_t(m_swapchainExtent.width), 0, 1 } };
      vk::ImageSubresourceLayers layers{ vk::ImageAspectFlags{vk::ImageAspectFlagBits::eColor}, 0, 0, 1 };
      vk::ImageBlit region {
        layers, srcoffsets,
        layers, dstoffsets
      };
      std::vector<vk::ImageBlit> regions = { region };

      vk::ImageMemoryBarrier swapchainTransferDstBarrier = {
        /*srcAccessMask=*/ vk::AccessFlagBits::eNone,
        /*dstAccessMask=*/ vk::AccessFlagBits::eTransferWrite,
        /*oldLayout=*/ vk::ImageLayout::eUndefined,
        /*newLayout=*/ vk::ImageLayout::eTransferDstOptimal,
        /*srcQueueFamilyIndex=*/ 0,
        /*dstQueueFamilyIndex=*/ 0,
        /*image=*/ m_swapchainImages[i],
        /*subresourceRange=*/ vk::ImageSubresourceRange( {vk::ImageAspectFlagBits::eColor, /*baseMipLevel=*/ 0, /*levelCount=*/ 1, /*baseArrayLayer=*/ 0, /*layerCount=*/ 1 })
      };

      b.pipelineBarrier(
        /*srcStageMask=*/ vk::PipelineStageFlagBits::eAllCommands,
        /*dstStageMask=*/ vk::PipelineStageFlagBits::eAllCommands,
        /*dependencyFlags=*/ vk::DependencyFlagBits::eByRegion,
        /*memory barriers=*/ nullptr,
        /*buffer memory barriers=*/ nullptr,
        /*image memory barriers=*/ swapchainTransferDstBarrier);

      b.blitImage(m_syncData[i].m_image.get(), vk::ImageLayout::eTransferSrcOptimal, m_swapchainImages[i], vk::ImageLayout::eTransferDstOptimal, vk::ArrayProxy<const vk::ImageBlit>{ 1, &region }, vk::Filter::eNearest);

      vk::ImageMemoryBarrier swapchainPresentSrcBarrier = {
        /*srcAccessMask=*/ vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
        /*dstAccessMask=*/ vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
        /*oldLayout=*/ vk::ImageLayout::eTransferDstOptimal,
        /*newLayout=*/ vk::ImageLayout::ePresentSrcKHR,
        /*srcQueueFamilyIndex=*/ 0,
        /*dstQueueFamilyIndex=*/ 0,
        /*image=*/ m_swapchainImages[i],
        /*subresourceRange=*/ vk::ImageSubresourceRange( {vk::ImageAspectFlagBits::eColor, /*baseMipLevel=*/ 0, /*levelCount=*/ 1, /*baseArrayLayer=*/ 0, /*layerCount=*/ 1 })
      };

      b.pipelineBarrier(
        /*srcStageMask=*/ vk::PipelineStageFlagBits::eAllCommands,
        /*dstStageMask=*/ vk::PipelineStageFlagBits::eAllCommands,
        /*dependencyFlags=*/ vk::DependencyFlagBits::eByRegion,
        /*memory barriers=*/ nullptr,
        /*buffer memory barriers=*/ nullptr,
        /*image memory barriers=*/ swapchainPresentSrcBarrier);

      b.end();
    }
  }

  // Window RT
  m_windowRenderTarget = new VKDirectSwapchainRenderTarget(this);
  m_windowRenderTarget->platformSetUpdatedWindowDimensions(surfaceWidth(), surfaceHeight());
  } catch (const std::exception& ex) {
    printf("%s\n", ex.what());
    abort();
  }
}

VKGLSyncData* RenderBackendVKDirect::acquireTexture() {
  if (m_firstFrame) {
    // don't wait for the first frame to be available - there's nobody to signal this
    m_firstFrame = false;
  } else {
    GLenum srcLayout = GL_LAYOUT_COLOR_ATTACHMENT_EXT;
    glWaitSemaphoreEXT(m_syncData[m_frameIndex].m_availableGL, 0, nullptr, 1, &m_syncData[m_frameIndex].m_textureGL, &srcLayout);
  }
  return &(m_syncData[m_frameIndex]);
}

void RenderBackendVKDirect::submitTexture(VKGLSyncData*) {
  // GL: signal rendering is done
  // VK: acquire image from swapchain
  // VK: blit texture to swapchain image (wait for GL finished, VK image acquired. signal VK blit done)
  // present (wait for VK blit done. signal VK image available)

  // signal GL is done
  GLenum targetLayout = GL_LAYOUT_TRANSFER_SRC_EXT;
  glSignalSemaphoreEXT(m_syncData[m_frameIndex].m_finishedGL, 0, nullptr, 1, &m_syncData[m_frameIndex].m_textureGL, &targetLayout);

  // RFE: handle return values
  auto r = m_device->acquireNextImageKHR(m_swapchain.get(), std::numeric_limits<uint64_t>::max(), m_imageAcquiredSemaphores[m_frameIndex].get(), vk::Fence());
  assert(m_frameIndex == r.value);  // this should be guaranteed, decoupling would mean N*M prepared blit command buffers

  // wait for GL finished & VK imageAcquired, blit/copy current texture onto current swapchain image, signal VK blit finished
  std::array<vk::Semaphore, 2> blitWaitSemaphores { m_syncData[m_frameIndex].m_finished.get(), m_imageAcquiredSemaphores[m_frameIndex].get() };
  std::array<vk::PipelineStageFlags, 2> blitWaitStages { vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTopOfPipe };

  std::array<vk::Semaphore, 1> blitSignalSemaphores {m_blitFinishedSemaphores[m_frameIndex].get()};

  vk::SubmitInfo submitInfo{blitWaitSemaphores,
                            blitWaitStages,
                            m_blitCommandBuffers[m_frameIndex],
                            blitSignalSemaphores
  };

  m_presentQueue.submit(submitInfo, vk::Fence{});

  // present
  std::vector<vk::Semaphore> presentWaitSemaphores{m_blitFinishedSemaphores[m_frameIndex].get()};
  //std::vector<vk::Semaphore> presentSignalSemaphores{m_syncData[m_frameIndex].m_available.get()};

  // VK_KHR_display
  // present on Direct Display output
  m_presentQueue.presentKHR({presentWaitSemaphores, m_swapchain.get(), m_frameIndex});

  m_frameIndex = (m_frameIndex + 1) % m_swapchainImages.size();

  // RFE
  glSignalSemaphoreEXT(m_syncData[m_frameIndex].m_availableGL, 0, nullptr, 0, nullptr, nullptr);
}

uint32_t RenderBackendVKDirect::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
  vk::PhysicalDeviceMemoryProperties memProperties = m_gpu.getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  throw std::runtime_error("failed to find suitable memory type!");
}

VKDirectSwapchainRenderTarget::VKDirectSwapchainRenderTarget(RenderBackendVKDirect* backend) : m_backend(backend) {

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
