#pragma once
#include "RenderBackend.h"
#include "rhi/gl/RHIWindowRenderTargetGL.h"
#include "rhi/vk/RHIVKFrameSource.h"
#include "rhi/vk/RHIVulkan.h" // pulls in vulkan.hpp with the dynamic-dispatch macro
#include <epoxy/egl.h>

#include <atomic>
#include <thread>
#include <vector>

class RenderBackendVKDirect;

class RenderBackendVKDirect : public RenderBackend, public VKFrameSource {
public:
  RenderBackendVKDirect();
  virtual ~RenderBackendVKDirect();

  virtual void createGLContext() override;
  virtual void createPresentation() override;

  virtual uint32_t surfaceWidth() const override { return m_swapchainExtent.width; }
  virtual uint32_t surfaceHeight() const override { return m_swapchainExtent.height; }
  virtual double refreshRateHz() const override { return m_refreshRateHz; }

  virtual EGLDisplay eglDisplay() const override { return m_eglDisplay; }
  virtual EGLContext eglContext() const override { return m_eglContext; }
  virtual EGLSurface eglSurface() const override { return EGL_NO_SURFACE; }
  virtual EGLConfig eglConfig() const override { return (EGLConfig) 0; }

  virtual RHIRenderTarget::ptr windowRenderTarget() const override { return m_windowRenderTarget; }

  virtual uint64_t lastPresentationTimestamp() const override { return m_lastPresentationTimestamp.load(std::memory_order_acquire); }

  // VKFrameSource: VK-native acquire/present.
  // RHIVK owns its own command buffer pool and records render commands; this backend hands out swap images + sync primitives.
  virtual uint32_t swapImageCount() const override { return static_cast<uint32_t>(m_swapchainImages.size()); }
  virtual VKFrameInfo acquireVKFrame() override;
  virtual void presentVKFrame(const VKFrameInfo&) override;

protected:
  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags);

  double m_refreshRateHz = 0;

  // EGL state
  EGLDisplay m_eglDisplay = EGL_NO_DISPLAY;
  EGLContext m_eglContext = NULL;

  // RHIWindowRenderTargetVK
  RHIRenderTarget::ptr m_windowRenderTarget;

  // VK state. Instance, physical device, logical device, queue family index,
  // and queue all live on rhi()->vk(); only display/surface/swapchain belong to this backend.
  struct Display {
    vk::DisplayKHR m_displayKHR;
    vk::DisplayPropertiesKHR m_displayProperties;
    vk::DisplayModePropertiesKHR m_modeProperties;
  };

  Display m_display;
  vk::UniqueSurfaceKHR m_surface;
  vk::UniqueSwapchainKHR m_swapchain;
  std::vector<vk::Image> m_swapchainImages;
  vk::Extent2D m_swapchainExtent;
  vk::Format m_swapchainFormat{vk::Format::eUndefined};
  uint32_t m_frameIndex = 0;
  std::vector<vk::UniqueSemaphore> m_imageAcquiredSemaphores;
  // Per-frame acquire fences. acquireVKFrame acquires the
  // swap image with a fence (not a semaphore) and host-waits it, so the render
  // submit never has to wait on imageAcquired. A semaphore-based acquire makes
  // the graphics queue wait on the presentation engine; once presentation
  // stops at shutdown the whole image pool can deadlock (all images acquired,
  // none freed), which loses the device. Indexed by m_frameIndex.
  std::vector<vk::UniqueFence> m_imageAcquiredFences;
  // Per-swap-image renderFinished semaphores.
  // Indexed by the swap image index returned by acquireNextImageKHR — not
  // by m_frameIndex. Per-image indexing is what makes the signal/wait pair
  // safe across reordered present modes (Immediate, Mailbox): when
  // acquireNextImageKHR returns image I, image I's previous present must
  // have already consumed renderFinished[I], so the next signal is safe.
  std::vector<vk::UniqueSemaphore> m_renderFinishedSemaphoresPerImage;

  friend class VKDirectSwapchainRenderTarget;

  uint32_t m_unsignaledFrames; // decremented each frame; skip available-semaphore wait while > 0

  // Presentation timestamp tracking
  std::atomic<uint64_t> m_lastPresentationTimestamp{0};
  std::thread m_scanoutThread;

#ifdef IS_TEGRA
  // NvRmHost1x syncpoint path. The worker waits on the nvkms-fence vblank syncpoint, captures the
  // kernel-recorded timestamp on each increment, and translates from the returned
  // CLOCK_MONOTONIC timebase into currentTimeNs()'s timebase (TSC-relative).
  // If the Host1x init fails, lastPresentationTimestamp will stay at 0.
  std::atomic<bool> m_scanoutShutdown{false};
  void* m_nvrmHost1x = nullptr; // NvRmHost1xOpen() session
  bool initScanoutSyncpt(); // returns true if the syncpt path is usable
#else
  // VK_EXT_display_control: scanout timestamp tracking via worker thread.
  // Atomic mailbox: main thread exchanges in a new fence, worker thread picks
  // it up. Stale fences displaced by new ones are destroyed by the producer.
  // This is less accurate than the NvRmHost1x path, but is platform-agnostic.
  // Mostly keeping this around in case Nvidia ever gets around to implementing
  // VK_GOOGLE_display_timing or similar.
  std::atomic<VkFence> m_scanoutFenceMailbox{VK_NULL_HANDLE};
  int m_scanoutEventFd = -1; // eventfd for waking the worker thread
#endif // IS_TEGRA

  void scanoutThreadFunc();
};
