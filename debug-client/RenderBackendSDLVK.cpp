#include "RenderBackendSDLVK.h"
#include "rhi/RHI.h"
#include "rhi/vk/RHIVK.h"
#include <SDL_vulkan.h>
#include <algorithm>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

namespace {

template <typename T>
bool contains(const std::vector<T>& container, const T& value) {
  return std::find(container.begin(), container.end(), value) != container.end();
}

} // namespace

RenderBackendSDLVK::RenderBackendSDLVK(SDL_Window* window) :
  m_window(window) {
}

RenderBackendSDLVK::~RenderBackendSDLVK() {
  // RHIVK is torn down separately by shutdownRHI(); by the time we get here
  // the device has been waitIdle'd by RHIVK's destructor (or it will be by
  // the time UniqueSwapchainKHR/UniqueSurfaceKHR destruction runs through
  // the dispatcher). Nothing to do here beyond letting the unique handles
  // drop.
}

/*static*/ bool RenderBackendSDLVK::getRequiredInstanceExtensions(SDL_Window* window, std::vector<const char*>& outExtensions) {
  unsigned int count = 0;
  if (!SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr)) {
    fprintf(stderr, "SDL_Vulkan_GetInstanceExtensions(count): %s\n", SDL_GetError());
    return false;
  }
  // SDL_Vulkan_GetInstanceExtensions returns pointers to strings owned by
  // SDL — safe to keep across the lifetime of the SDL library.
  outExtensions.resize(count);
  if (!SDL_Vulkan_GetInstanceExtensions(window, &count, outExtensions.data())) {
    fprintf(stderr, "SDL_Vulkan_GetInstanceExtensions(names): %s\n", SDL_GetError());
    return false;
  }
  return true;
}

void RenderBackendSDLVK::createPresentation() {
  vk::Instance instance = rhi()->vk()->instance();

  // Create the VkSurfaceKHR via SDL; this picks the right platform extension
  // (xlib / xcb / wayland) under the hood.
  VkSurfaceKHR rawSurface = VK_NULL_HANDLE;
  if (!SDL_Vulkan_CreateSurface(m_window, instance, &rawSurface)) {
    fprintf(stderr, "SDL_Vulkan_CreateSurface failed: %s\n", SDL_GetError());
    abort();
  }
  // Wrap the raw VkSurfaceKHR in vk::UniqueSurfaceKHR so the destructor
  // calls vkDestroySurfaceKHR for us. The deleter needs the instance handle.
  m_surface = vk::UniqueSurfaceKHR(vk::SurfaceKHR(rawSurface),
    vk::ObjectDestroy<vk::Instance, VULKAN_HPP_DEFAULT_DISPATCHER_TYPE>(instance, nullptr, VULKAN_HPP_DEFAULT_DISPATCHER));

  // Verify the queue family selected by RHIVulkan supports presentation on
  // this surface.
  vk::PhysicalDevice gpu = rhi()->vk()->physicalDevice();
  uint32_t queueFamily = rhi()->vk()->queueFamilyIndex();
  if (!gpu.getSurfaceSupportKHR(queueFamily, m_surface.get())) {
    fprintf(stderr, "RenderBackendSDLVK: shared queue family %u does not support presentation on the SDL surface\n", queueFamily);
    abort();
  }

  buildSwapchainAndSemaphores();

  // Initial render target. recreateSwapchain mutates this in place via
  // platformSetUpdatedWindowParameters.
  m_windowRenderTarget = new RHIWindowRenderTargetVK(
    m_swapchainExtent.width, m_swapchainExtent.height, m_swapchainFormat);
}

void RenderBackendSDLVK::buildSwapchainAndSemaphores() {
  vk::Device device = rhi()->vk()->device();
  vk::PhysicalDevice gpu = rhi()->vk()->physicalDevice();

  auto formats = gpu.getSurfaceFormatsKHR(m_surface.get());
  auto capabilities = gpu.getSurfaceCapabilitiesKHR(m_surface.get());
  auto presentModes = gpu.getSurfacePresentModesKHR(m_surface.get());

  // Pick a sane format. Prefer BGRA8 (most common surface format on X11/
  // Wayland with the Mesa stack); fall back to RGBA8, then whatever the
  // surface offers first.
  vk::SurfaceFormatKHR preferredFormats[] = {
    {      vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear},
    {      vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear},
    {vk::Format::eA8B8G8R8UnormPack32, vk::ColorSpaceKHR::eSrgbNonlinear},
  };
  vk::SurfaceFormatKHR format = formats[0];
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

  // SDL window may have a different drawable size than what we requested
  // (HiDPI / DPR > 1). Use SDL_Vulkan_GetDrawableSize to ask for the actual
  // pixel-space dimensions, but defer to capabilities.currentExtent when it
  // is reported as a fixed value (not 0xFFFFFFFF).
  vk::Extent2D extent;
  if (capabilities.currentExtent.width != 0xFFFFFFFF) {
    extent = capabilities.currentExtent;
  } else {
    int w = 0, h = 0;
    SDL_Vulkan_GetDrawableSize(m_window, &w, &h);
    extent.width = std::clamp<uint32_t>(uint32_t(w), capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent.height = std::clamp<uint32_t>(uint32_t(h), capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
  }

  // Mailbox > Immediate > Fifo. Fifo is the always-supported fallback.
  vk::PresentModeKHR presentMode = vk::PresentModeKHR::eFifo;
  if (contains(presentModes, vk::PresentModeKHR::eMailbox)) {
    presentMode = vk::PresentModeKHR::eMailbox;
  } else if (contains(presentModes, vk::PresentModeKHR::eImmediate)) {
    presentMode = vk::PresentModeKHR::eImmediate;
  }
  if (const char* envMode = getenv("RHI_VK_PRESENT_MODE")) {
    if (strcmp(envMode, "fifo") == 0)
      presentMode = vk::PresentModeKHR::eFifo;
    else if (strcmp(envMode, "mailbox") == 0 && contains(presentModes, vk::PresentModeKHR::eMailbox))
      presentMode = vk::PresentModeKHR::eMailbox;
    else if (strcmp(envMode, "immediate") == 0 && contains(presentModes, vk::PresentModeKHR::eImmediate))
      presentMode = vk::PresentModeKHR::eImmediate;
  }

  uint32_t imageCount = capabilities.minImageCount + 1;
  if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
    imageCount = capabilities.maxImageCount;

  vk::SurfaceTransformFlagBitsKHR pretransform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
  if ((pretransform & capabilities.supportedTransforms) != pretransform)
    pretransform = capabilities.currentTransform;

  // Pass the existing swapchain (if any) as oldSwapchain so the driver can
  // recycle resources. The UniqueSwapchainKHR destructor cleans it up after
  // we reassign m_swapchain.
  vk::SwapchainKHR oldSwapchain = m_swapchain.get();
  vk::SwapchainCreateInfoKHR ci{
    vk::SwapchainCreateFlagsKHR(),
    m_surface.get(),
    imageCount,
    format.format,
    format.colorSpace,
    extent,
    /*arrayLayers=*/ 1,
    vk::ImageUsageFlags(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst),
    vk::SharingMode::eExclusive,
    /*queueFamilyIndexCount=*/ 0, /*pQueueFamilyIndices=*/ nullptr,
    pretransform,
    vk::CompositeAlphaFlagBitsKHR::eOpaque,
    presentMode,
    /*clipped=*/ VK_TRUE,
    oldSwapchain};

  vk::UniqueSwapchainKHR newSwapchain = device.createSwapchainKHRUnique(ci);
  // Release the old swapchain only after the new one is created so the
  // driver can use oldSwapchain for recycling.
  m_swapchain = std::move(newSwapchain);

  m_swapchainImages = device.getSwapchainImagesKHR(m_swapchain.get());
  m_swapchainExtent = extent;
  m_swapchainFormat = format.format;

  // Rebuild per-image semaphores. The acquire ring is sized to swap image
  // count so the semaphore-reuse pattern matches RenderBackendVKDirect.
  m_imageAcquiredSemaphores.clear();
  m_renderFinishedSemaphoresPerImage.clear();
  for (size_t i = 0; i < m_swapchainImages.size(); ++i) {
    vk::SemaphoreCreateInfo sci{};
    m_imageAcquiredSemaphores.push_back(device.createSemaphoreUnique(sci));
    m_renderFinishedSemaphoresPerImage.push_back(device.createSemaphoreUnique(sci));
  }
  m_frameIndex = 0;

  printf("RenderBackendSDLVK: swapchain %ux%u, %s, %zu images, present mode %s\n",
    m_swapchainExtent.width, m_swapchainExtent.height,
    vk::to_string(m_swapchainFormat).c_str(),
    m_swapchainImages.size(),
    vk::to_string(presentMode).c_str());
}

void RenderBackendSDLVK::recreateSwapchain() {
  // Drain in-flight GPU work referencing the old swapchain. Required before
  // destroying the swap-image views cached by RHIVK + any per-image
  // semaphores we hold.
  rhi()->vk()->device().waitIdle();

  // Drop RHIVK's swap-image-view cache so it doesn't dereference stale
  // VkImage handles after the swapchain recreation.
  static_cast<RHIVK*>(rhi())->invalidateSwapchainResources();

  buildSwapchainAndSemaphores();

  if (m_windowRenderTarget) {
    m_windowRenderTarget->platformSetUpdatedWindowParameters(
      m_swapchainExtent.width, m_swapchainExtent.height, m_swapchainFormat);
  }
}

VKFrameInfo RenderBackendSDLVK::acquireVKFrame() {
  vk::Device device = rhi()->vk()->device();

  // Acquire (and rebuild + retry on out-of-date). Up to two iterations: the
  // first might come back OUT_OF_DATE if the WM resize raced the previous
  // frame; the second retry should hit a fresh swapchain.
  for (int attempt = 0; attempt < 2; ++attempt) {
    vk::Semaphore imageAcquired = m_imageAcquiredSemaphores[m_frameIndex].get();
    vk::Result result;
    uint32_t swapchainIndex = 0;
    // Use the raw C entry point to avoid vulkan-hpp's auto-throw on
    // OUT_OF_DATE — we want to inspect the result and recreate inline.
    VkResult cr = VULKAN_HPP_DEFAULT_DISPATCHER.vkAcquireNextImageKHR(
      device, m_swapchain.get(), std::numeric_limits<uint64_t>::max(),
      imageAcquired, VK_NULL_HANDLE, &swapchainIndex);
    result = static_cast<vk::Result>(cr);

    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapchain();
      continue;
    }
    if (result == vk::Result::eSuboptimalKHR)
      fprintf(stderr, "RenderBackendSDLVK::acquireVKFrame: eSuboptimalKHR\n");
    else if (result != vk::Result::eSuccess) {
      fprintf(stderr, "RenderBackendSDLVK::acquireVKFrame: vkAcquireNextImageKHR returned %s\n", vk::to_string(result).c_str());
      abort();
    }

    vk::Semaphore renderFinished = m_renderFinishedSemaphoresPerImage[swapchainIndex].get();
    VKFrameInfo frame;
    frame.swapchainIndex = swapchainIndex;
    frame.swapchainImage = m_swapchainImages[swapchainIndex];
    frame.extent = m_swapchainExtent;
    frame.format = m_swapchainFormat;
    frame.imageAcquired = imageAcquired;
    frame.renderFinished = renderFinished;
    return frame;
  }
  fprintf(stderr, "RenderBackendSDLVK::acquireVKFrame: failed to acquire after swapchain recreate\n");
  abort();
}

void RenderBackendSDLVK::presentVKFrame(const VKFrameInfo& frame) {
  vk::Queue presentQueue = rhi()->vk()->queue();

  vk::SwapchainKHR swapchain = m_swapchain.get();
  vk::Semaphore waitSem = frame.renderFinished;
  uint32_t swapIdx = frame.swapchainIndex;
  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = reinterpret_cast<const VkSemaphore*>(&waitSem);
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = reinterpret_cast<const VkSwapchainKHR*>(&swapchain);
  presentInfo.pImageIndices = &swapIdx;
  // Raw C entry point to avoid vulkan-hpp's auto-throw on OUT_OF_DATE; we
  // want to recreate the swapchain inline and continue.
  VkResult cr = VULKAN_HPP_DEFAULT_DISPATCHER.vkQueuePresentKHR(presentQueue, &presentInfo);
  vk::Result presentResult = static_cast<vk::Result>(cr);
  if (presentResult == vk::Result::eErrorOutOfDateKHR) {
    // Window was resized between acquire and present (or between present
    // and acquire on the previous frame). The frame's pixels are lost,
    // which is fine during resize. Rebuild for the next acquire.
    recreateSwapchain();
  } else if (presentResult == vk::Result::eSuboptimalKHR) {
    fprintf(stderr, "RenderBackendSDLVK::presentVKFrame: eSuboptimalKHR\n");
  } else if (presentResult != vk::Result::eSuccess) {
    fprintf(stderr, "RenderBackendSDLVK::presentVKFrame: vkQueuePresentKHR returned %s\n", vk::to_string(presentResult).c_str());
    abort();
  }

  m_frameIndex = (m_frameIndex + 1) % m_swapchainImages.size();
}
