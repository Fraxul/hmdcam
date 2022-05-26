#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "RenderBackend.h"
#include <epoxy/egl.h>
#include "rhi/gl/RHIWindowRenderTargetGL.h"
#include "vulkan/vulkan.hpp"

#include <vector>

class RenderBackendVKDirect;

struct VKGLSyncData {
  // VK texture
  vk::UniqueImage         m_image;
  vk::UniqueDeviceMemory  m_deviceMemory;
  int                     m_handle;
  GLuint                  m_memoryObject;

  // GL texture
  GLuint m_textureGL;
  GLuint m_framebufferGL;
  GLuint m_internalFormat;

  // VK semaphores
  vk::UniqueSemaphore m_available; // signal when image is available
  vk::UniqueSemaphore m_finished;  // wait for GL to be finished
  int                 m_availableHandle;
  int                 m_finishedHandle;

  // GL semaphores
  GLuint              m_availableGL;
  GLuint              m_finishedGL;
};

class VKDirectSwapchainRenderTarget : public RHIWindowRenderTargetGL {
public:
  typedef boost::intrusive_ptr<VKDirectSwapchainRenderTarget> ptr;
  VKDirectSwapchainRenderTarget(RenderBackendVKDirect* backend);
  virtual ~VKDirectSwapchainRenderTarget();
  virtual void platformSwapBuffers();

protected:
  void internalAcquireTexture();

  RenderBackendVKDirect* m_backend;
  VKGLSyncData* m_syncTex = NULL;
};

class RenderBackendVKDirect : public RenderBackend {
public:
  RenderBackendVKDirect();
  virtual ~RenderBackendVKDirect() {}

  virtual void init();

  virtual uint32_t surfaceWidth() const { return m_swapchainExtent.width; }
  virtual uint32_t surfaceHeight() const { return m_swapchainExtent.height; }
  virtual double refreshRateHz() const { return m_refreshRateHz; }

  virtual EGLDisplay eglDisplay() const { return m_eglDisplay; }
  virtual EGLContext eglContext() const { return m_eglContext; }
  virtual EGLSurface eglSurface() const { return EGL_NO_SURFACE; }
  virtual EGLConfig eglConfig() const { return (EGLConfig) 0; }

  virtual RHIRenderTarget::ptr windowRenderTarget() const { return m_windowRenderTarget; }

protected:

  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags);

  double m_refreshRateHz = 0;

  // EGL state
  int m_drmFd = -1;
  EGLDeviceEXT m_eglDevice = NULL;
  EGLDisplay m_eglDisplay = EGL_NO_DISPLAY;
  EGLContext m_eglContext = NULL;

  VKDirectSwapchainRenderTarget::ptr m_windowRenderTarget;

  // VK state
  vk::DynamicLoader m_dl;

  struct Display {
    vk::DisplayKHR m_displayKHR;
    vk::DisplayPropertiesKHR m_displayProperties;
    vk::DisplayModePropertiesKHR m_modeProperties;
  };


  vk::UniqueInstance m_instance;
  vk::PhysicalDevice m_gpu;
  Display m_display;
  vk::UniqueSurfaceKHR m_surface;
  uint32_t m_presentFamily = 0;
  vk::Queue m_presentQueue;
  vk::UniqueDevice m_device;
  vk::UniqueSwapchainKHR m_swapchain;
  std::vector<vk::Image> m_swapchainImages;
  vk::Extent2D m_swapchainExtent;
  vk::Format m_swapchainFormat{ vk::Format::eUndefined };
  uint32_t m_frameIndex = 0;
  std::vector<VKGLSyncData> m_syncData;
  std::vector<vk::UniqueSemaphore> m_imageAcquiredSemaphores;
  std::vector<vk::UniqueSemaphore> m_blitFinishedSemaphores;
  vk::CommandPool m_commandPool;
  std::vector<vk::CommandBuffer>  m_blitCommandBuffers;

  friend class VKDirectSwapchainRenderTarget;
  VKGLSyncData* acquireTexture();
  void submitTexture(VKGLSyncData*);

  bool m_firstFrame = true;
};

