#pragma once
#include "rhi/RHIObject.h"
#include <cstdint>

// RHIFence: a CPU-waitable fence that becomes signaled when a unit of GPU work
// completes. Created via RHI::registerFrameCompletionFence() (see RHI.h); the
// backend arranges for it to signal at the next queue submission.
//
// This is the host-synchronization analogue of RHIQuery -- the simplest kind
// of RHI object, owning a single backend sync primitive (a VkFence on the VK
// backend) and exposing only blocking-wait / poll.
//
// Typical use is cross-thread handoff: the render thread registers a fence
// after recording the work that fills a shared surface, hands the fence to a
// consumer thread, and that thread blocks on wait() before reading the surface
// on another engine (e.g. the Tegra VIC).
class RHIFence : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIFence> ptr;
  virtual ~RHIFence();

  // Block the calling thread until the fence is signaled or timeoutNs elapses.
  // Returns true if the fence signaled, false on timeout. The default
  // UINT64_MAX waits indefinitely.
  virtual bool wait(uint64_t timeoutNs = UINT64_MAX) = 0;

  // Non-blocking poll. Returns true if the fence is already signaled.
  virtual bool isSignaled() = 0;

protected:
  RHIFence();
};
