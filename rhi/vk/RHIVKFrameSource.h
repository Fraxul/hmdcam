#pragma once
// VKFrameSource: abstract interface that supplies swap-chain frames to RHIVK.
//
// Decouples RHIVK from the concrete render backend so the RHI VK library can
// be lifted into projects with a different presentation path.

#include "rhi/vk/RHIVulkan.h" // for vk::* types

struct VKFrameInfo {
  uint32_t swapchainIndex;
  vk::Image swapchainImage;
  vk::Extent2D extent;
  vk::Format format;
  // imageAcquired: signaled when the swap image is ready to render into.
  //   The caller's render submit must wait on this with a
  //   color-attachment-output stage mask.
  // renderFinished: the caller's render submit must signal this; the
  //   subsequent present waits on it.
  vk::Semaphore imageAcquired;
  vk::Semaphore renderFinished;
};

class VKFrameSource {
public:
  virtual ~VKFrameSource() = default;

  // Number of swap images. Sizes RHIVK's per-frame ring of command buffers
  // and fences.
  virtual uint32_t swapImageCount() const = 0;

  // Acquire the next swap image. Returns its VkImage handle + per-frame
  // semaphores. Blocks if no image is available (vkAcquireNextImageKHR
  // semantics).
  virtual VKFrameInfo acquireVKFrame() = 0;

  // Present the frame previously returned by acquireVKFrame.
  virtual void presentVKFrame(const VKFrameInfo&) = 0;
};
