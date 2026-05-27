#pragma once
// RHIInteropBufferVK: a VkBuffer + CUDA device pointer backed by the same
// Vulkan-allocated exportable memory.
//
// VK side is a regular RHIBufferVK — bindable via vertex/index/indirect
// commands through the standard RHI APIs. CUDA side gets a mapped device
// pointer via cudaImportExternalMemory + cudaExternalMemoryGetMappedBuffer.
// The same physical memory is visible to both APIs; RHIInteropSync handles
// the producer/consumer handshake.
//
// Replaces the GL-counterpart RHIInteropBufferGL for the VK RHI backend.

#include "rhi/RHIInteropBuffer.h"
#include "rhi/vk/RHIBufferVK.h"
#include "rhi/vk/RHIInteropSyncDescriptor.h"
#include <cuda.h>
#include <cuda_runtime.h>

class RHIInteropBufferVK : public RHIBufferVK, public RHIInteropBuffer {
public:
  typedef boost::intrusive_ptr<RHIInteropBufferVK> ptr;

  static RHIInteropBufferVK* newBuffer(size_t sizeBytes, RHIBufferUsageMode, const RHIInteropSyncDescriptor&);

  virtual ~RHIInteropBufferVK();

  CUdeviceptr cudaPointer() const override { return m_cudaPtr; }
  size_t cudaSize() const override { return m_cudaSize; }

  const RHIInteropSyncDescriptor& syncDescriptor() const { return m_syncDescriptor; }

private:
  friend class RHIInteropSync;
  RHIInteropBufferVK();

  cudaExternalMemory_t m_cudaExtMem = nullptr;
  CUdeviceptr m_cudaPtr = 0;
  size_t m_cudaSize = 0;
  RHIInteropSyncDescriptor m_syncDescriptor;
};
