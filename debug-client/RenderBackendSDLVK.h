#pragma once
// VK presentation backend for debug-client. Owns an SDL_Window, a
// VkSurfaceKHR, and a VkSwapchainKHR; bridges to RHIVK via VKFrameSource.
// Supports swapchain recreation on window resize.
//
// Lives in debug-client/ (not rhi/) because it pulls in SDL and is specific
// to the windowed-desktop presentation path; the VK RHI itself stays
// portable across host backends.

#include "rhi/vk/RHIVKFrameSource.h"
#include "rhi/vk/RHIVulkan.h"
#include "rhi/vk/RHIWindowRenderTargetVK.h"
#include <SDL.h>
#include <vector>

class RenderBackendSDLVK : public VKFrameSource {
public:
  RenderBackendSDLVK(SDL_Window* window);
  virtual ~RenderBackendSDLVK();

  // Query SDL for the instance extensions it requires for VkSurfaceKHR
  // creation on the current platform. Caller passes these into initRHIVulkan.
  // Returns false on failure (e.g. SDL Vulkan loader didn't initialize).
  static bool getRequiredInstanceExtensions(SDL_Window* window, std::vector<const char*>& outExtensions);

  // After rhi()->vk() has been initialized, create the VkSurfaceKHR (via
  // SDL_Vulkan_CreateSurface) and the initial swapchain. The owned
  // RHIWindowRenderTargetVK becomes valid here.
  void createPresentation();

  RHIRenderTarget::ptr windowRenderTarget() const { return m_windowRenderTarget; }

  // Recreate the swapchain to match the current SDL window size. Drains the
  // device, destroys swap-image-tied resources (including RHIVK's view
  // cache), allocates a new swapchain + per-image semaphores, then updates
  // the RHIWindowRenderTargetVK's reported dimensions.
  void recreateSwapchain();

  // VKFrameSource
  virtual uint32_t swapImageCount() const override { return static_cast<uint32_t>(m_swapchainImages.size()); }
  virtual VKFrameInfo acquireVKFrame() override;
  virtual void presentVKFrame(const VKFrameInfo&) override;

protected:
  SDL_Window* m_window = nullptr;
  vk::UniqueSurfaceKHR m_surface;

  vk::UniqueSwapchainKHR m_swapchain;
  std::vector<vk::Image> m_swapchainImages;
  vk::Extent2D m_swapchainExtent{};
  vk::Format m_swapchainFormat{vk::Format::eUndefined};

  // Acquire semaphores are indexed by the per-frame slot index (rolls
  // 0..swapImageCount-1). Each frame draws an image and signals its image-
  // acquired semaphore; the next acquire uses the next slot's semaphore.
  // renderFinished is per-swap-image (indexed by acquireNextImageKHR's
  // returned index): matches the pattern in RenderBackendVKDirect — safe
  // across reordered present modes since image I's previous present must
  // have already consumed renderFinished[I].
  std::vector<vk::UniqueSemaphore> m_imageAcquiredSemaphores;
  std::vector<vk::UniqueSemaphore> m_renderFinishedSemaphoresPerImage;
  uint32_t m_frameIndex = 0;

  RHIWindowRenderTargetVK::ptr m_windowRenderTarget;

  // Build (or rebuild) the swapchain + per-image semaphores. Used by both
  // first-time createPresentation and recreateSwapchain.
  void buildSwapchainAndSemaphores();
};
