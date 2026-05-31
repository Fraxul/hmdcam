#pragma once
// RHIFenceVK: VK-backend RHIFence backed by a timeline-semaphore target value.
//
// Holds a *borrowed* handle to RHIVK's long-lived frame-completion timeline
// semaphore plus the value that semaphore reaches when the registering frame's
// GPU work completes (see RHIVK::registerFrameCompletionFence / swapBuffers /
// flush, which signal ++m_frameCompletionValue per submit). wait() is a
// host-side vkWaitSemaphores on that value.
//
// Crucially, this object owns no Vulkan resource: the timeline is owned by
// RHIVK for the device's lifetime. A consumer (e.g. the NvEnc submit worker)
// can therefore time out and drop its RHIFence without any risk of destroying a
// sync object a queue submission still references -- the failure mode that a
// per-frame VkFence + signal-only submit suffered (VUID-vkDestroyFence-01120).

#include "rhi/RHIFence.h"
#include "rhi/vk/RHIVulkan.h"

class RHIFenceVK : public RHIFence {
public:
  typedef boost::intrusive_ptr<RHIFenceVK> ptr;

  // timeline is borrowed (owned by RHIVK). Signaled when timeline >= targetValue.
  RHIFenceVK(vk::Semaphore timeline, uint64_t targetValue);
  virtual ~RHIFenceVK();

  virtual bool wait(uint64_t timeoutNs) override;
  virtual bool isSignaled() override;

protected:
  vk::Semaphore m_timeline; // borrowed, not owned
  uint64_t m_targetValue = 0;
};
