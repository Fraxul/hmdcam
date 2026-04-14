#pragma once
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include "RenderBackend.h"
#include <epoxy/egl.h>
#include "rhi/gl/RHIWindowRenderTargetGL.h"

// vulkan.hpp library triggers a bunch of -Wshadow warnings -- just ignore them since it's third-party code
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#include "vulkan/vulkan.hpp"
#pragma clang diagnostic pop

#include <atomic>
#include <thread>
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
  virtual ~RenderBackendVKDirect();

  virtual void init();

  virtual uint32_t surfaceWidth() const { return m_swapchainExtent.width; }
  virtual uint32_t surfaceHeight() const { return m_swapchainExtent.height; }
  virtual double refreshRateHz() const { return m_refreshRateHz; }

  virtual EGLDisplay eglDisplay() const { return m_eglDisplay; }
  virtual EGLContext eglContext() const { return m_eglContext; }
  virtual EGLSurface eglSurface() const { return EGL_NO_SURFACE; }
  virtual EGLConfig eglConfig() const { return (EGLConfig) 0; }

  virtual RHIRenderTarget::ptr windowRenderTarget() const { return m_windowRenderTarget; }

  virtual uint64_t lastPresentationTimestamp() const { return m_lastPresentationTimestamp.load(std::memory_order_acquire); }

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

  uint32_t m_unsignaledFrames; // decremented each frame; skip available-semaphore wait while > 0

  // VK_EXT_display_control: scanout timestamp tracking via worker thread.
  // Atomic mailbox: main thread exchanges in a new fence, worker thread picks
  // it up. Stale fences displaced by new ones are destroyed by the producer.
  std::atomic<uint64_t> m_lastPresentationTimestamp{0};
  std::thread m_scanoutThread;
  std::atomic<VkFence> m_scanoutFenceMailbox{VK_NULL_HANDLE};
  int m_scanoutEventFd = -1; // eventfd for waking the worker thread
  void scanoutThreadFunc();
};

