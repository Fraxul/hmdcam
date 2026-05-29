#pragma once
// RHIInteropSync: Timeline-semaphore CUDA↔VK synchronization barrier. A single
// VK timeline semaphore is exported via opaque-FD and imported into CUDA so
// both APIs share one monotonic counter. Each side bumps the counter when it
// signals; the other side waits on the most-recently-signaled value.
#include "rhi/RHIObject.h"
#include "rhi/vk/RHIVulkan.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

class RHIInteropSync : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIInteropSync> ptr;
  RHIInteropSync();
  virtual ~RHIInteropSync();

  // Sync barrier functions.

  // Call this when you're done with CUDA work and will be switching to RHI.
  // Bumps the shared counter and records a CUDA-side signal on the stream.
  // The RHI consumer (RHIVK's per-frame submit) waits on the latest value.
  void signalCUDAToRHI(CUstream stream);

  // Call this when you're done with RHI work and will be switching to CUDA.
  // Records a CUDA-side wait on the latest VK-signaled value. The VK side
  // signals every frame from RHIVK's swap/flush submit, so this is correct
  // as long as the VK submit that produced the data has already happened.
  void signalRHIToCUDA(CUstream stream);

  // RHIVK accessors: read by the per-frame submit to populate
  // VkTimelineSemaphoreSubmitInfo. The VK wait value is whatever CUDA most
  // recently signaled; the VK signal value is allocated here so a single
  // monotonic counter is shared across both directions.
  vk::Semaphore vkSemaphore() const { return m_timeline.semaphore.get(); }
  // VK's per-frame submit waits on the most recent value CUDA signaled.
  uint64_t cudaSignaledValue() const { return m_cudaSignaledValue; }
  // Allocate the next value for a VK signal. Drawn from the single shared
  // counter (see m_nextValue) so VK- and CUDA-signaled values never collide.
  uint64_t allocateVKSignalValue() { return (m_vkSignaledValue = ++m_nextValue); }

protected:
  RHIVulkan::ExternalSemaphore m_timeline;
  cudaExternalSemaphore_t m_timelineCU = nullptr;

  // Single monotonic value allocator shared by both producers. Every signal consumes
  // the next value from here, so no two signals ever target the same timeline
  // value, and a given wait value can only be reached by its intended producer.
  uint64_t m_nextValue = 0;
  // Last value signaled by CUDA (set in signalCUDAToRHI). VK waits on this.
  uint64_t m_cudaSignaledValue = 0;
  // Last value signaled by VK (set in allocateVKSignalValue). CUDA (signalRHIToCUDA) waits on this.
  uint64_t m_vkSignaledValue = 0;
};
