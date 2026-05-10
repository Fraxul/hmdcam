#pragma once
// RHIInteropBufferGL: a GL buffer and a CUDA device pointer backed by the same Vulkan-allocated VkBuffer.

#include "rhi/vk/RHIInteropSyncDescriptor.h"
#include "rhi/gl/RHIBufferGL.h"
#include "rhi/vk/RHIVulkan.h"
#include <cuda.h>
#include <cuda_runtime.h>

class RHIInteropBufferGL : public RHIBufferGL {
public:
  typedef boost::intrusive_ptr<RHIInteropBufferGL> ptr;

  static RHIInteropBufferGL* newBuffer(size_t sizeBytes, RHIBufferUsageMode, const RHIInteropSyncDescriptor&);

  virtual ~RHIInteropBufferGL();

  CUdeviceptr cudaPointer() const { return m_cudaPtr; }
  size_t cudaSize() const { return m_cudaSize; }

  const RHIInteropSyncDescriptor& syncDescriptor() const { return m_syncDescriptor; };

private:
  friend class RHIInteropSync;
  RHIInteropBufferGL();

  RHIVulkan::ExternalBuffer m_vkBuffer;
  GLuint m_glMemoryObject = 0;

  cudaExternalMemory_t m_cudaExtMem = nullptr;
  CUdeviceptr m_cudaPtr = 0;
  size_t m_cudaSize = 0;
  RHIInteropSyncDescriptor m_syncDescriptor;
};
