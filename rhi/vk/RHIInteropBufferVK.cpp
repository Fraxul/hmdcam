#include "rhi/vk/RHIInteropBufferVK.h"
#include "rhi/RHI.h"
#include "rhi/cuda/CudaUtil.h"
#include <stdio.h>
#include <unistd.h>

RHIInteropBufferVK::RHIInteropBufferVK() :
  RHIBufferVK() {
  // Actual init happens in RHIInteropBufferVK::newBuffer().
}

RHIInteropBufferVK::~RHIInteropBufferVK() {
  if (m_cudaPointer) {
    cuMemFree(m_cudaPointer);
    m_cudaPointer = 0;
  }
  if (m_cudaExtMem) {
    cudaDestroyExternalMemory(m_cudaExtMem);
    m_cudaExtMem = nullptr;
  }
  // Base RHIBufferVK destructor releases m_buffer / m_memory (UniqueXxx).
}

/*static*/ RHIInteropBufferVK* RHIInteropBufferVK::newBuffer(size_t sizeBytes, RHIBufferUsageMode usageMode) {
  RHIVulkan* vk = rhi()->vk();
  if (!vk) {
    fprintf(stderr, "RHIInteropBufferVK::newBuffer: rhi()->vk() is null; cannot allocate interop buffer\n");
    abort();
  }

  RHIInteropBufferVK* buf = new RHIInteropBufferVK();
  buf->m_size = sizeBytes;
  buf->m_usageMode = usageMode;

  // Permissive usage so the buffer can stand in anywhere the VK draw API
  // chooses to bind it. CUDA accesses the underlying memory directly
  // through the imported mapping, independent of VkBufferUsage.
  vk::BufferUsageFlags vkUsage =
    vk::BufferUsageFlagBits::eTransferSrc |
    vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer |
    vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;
  RHIVulkan::ExternalBuffer external = vk->allocateExternalBuffer(sizeBytes, vkUsage);

  // Move the externally-allocated VkBuffer + VkDeviceMemory into the base
  // RHIBufferVK fields so the standard RHIBufferVK::vkBuffer() accessor
  // (used by bindStreamBuffer / bindIndexBuffer / etc.) returns the same
  // handle the CUDA-imported memory backs.
  buf->m_buffer = std::move(external.buffer);
  buf->m_memory = std::move(external.memory);

  // CUDA: import the exported memory FD, then map a device pointer
  // covering the requested logical size (allocationSize may be padded by
  // the driver; we map only the user-requested range).
  cudaExternalMemoryHandleDesc memDesc = {};
  memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  memDesc.handle.fd = external.memoryFd;
  memDesc.size = external.allocationSize;
  CUDA_CHECK(cudaImportExternalMemory(&buf->m_cudaExtMem, &memDesc));
  // cudaImportExternalMemory takes ownership of the FD on success — null
  // the field to make the post-import double-close a clear error if some
  // future change mishandles the struct.
  external.memoryFd = -1;

  cudaExternalMemoryBufferDesc bufDesc = {};
  bufDesc.offset = 0;
  bufDesc.size = sizeBytes;
  bufDesc.flags = 0;
  void* cudaPtrRaw = nullptr;
  CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&cudaPtrRaw, buf->m_cudaExtMem, &bufDesc));
  buf->m_cudaPointer = reinterpret_cast<CUdeviceptr>(cudaPtrRaw);
  buf->m_isInteropBuffer = true;

  return buf;
}
