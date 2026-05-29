#pragma once
// RHIInteropBufferVK: a VkBuffer + CUDA device pointer backed by the same
// Vulkan-allocated exportable memory.
//
// VK side is a regular RHIBufferVK — bindable via vertex/index/indirect
// commands through the standard RHI APIs. CUDA side gets a mapped device
// pointer via cudaImportExternalMemory + cudaExternalMemoryGetMappedBuffer.
// The same physical memory is visible to both APIs.

#include "rhi/vk/RHIBufferVK.h"
#include <cuda.h>
#include <cuda_runtime.h>

class RHIInteropBufferVK : public RHIBufferVK {
public:
  typedef boost::intrusive_ptr<RHIInteropBufferVK> ptr;

  static RHIInteropBufferVK* newBuffer(size_t sizeBytes, RHIBufferUsageMode);

  virtual ~RHIInteropBufferVK();

private:
  RHIInteropBufferVK();

  cudaExternalMemory_t m_cudaExtMem = nullptr;
};
