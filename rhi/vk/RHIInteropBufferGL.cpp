#include "rhi/vk/RHIInteropBufferGL.h"
#include "rhi/RHI.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/gl/GLCommon.h"
#include <stdio.h>
#include <unistd.h>

RHIInteropBufferGL::RHIInteropBufferGL() :
  RHIBufferGL() {
  // Actual init happens in RHIInteropBufferGL::newBuffer()
}

RHIInteropBufferGL::~RHIInteropBufferGL() {
  if (m_syncDescriptor.sync())
    m_syncDescriptor.sync()->removeSyncedObject(this);

  if (m_cudaPtr) {
    cuMemFree(m_cudaPtr);
    m_cudaPtr = 0;
  }
  if (m_cudaExtMem) {
    cudaDestroyExternalMemory(m_cudaExtMem);
    m_cudaExtMem = nullptr;
  }
  if (m_glMemoryObject) {
    glDeleteMemoryObjectsEXT(1, &m_glMemoryObject);
    m_glMemoryObject = 0;
  }
  // ~RHIBufferGL deletes m_buffer.
}

/*static*/ RHIInteropBufferGL* RHIInteropBufferGL::newBuffer(size_t sizeBytes, RHIBufferUsageMode usageMode, const RHIInteropSyncDescriptor& syncDescriptor) {
  RHIVulkan* vk = rhi()->vk();
  if (!vk) {
    fprintf(stderr, "RHIInteropBufferGL::newBuffer: rhi()->vk() is null; cannot allocate interop buffer\n");
    abort();
  }

  RHIInteropBufferGL* buf = new RHIInteropBufferGL();
  buf->m_size = sizeBytes;
  buf->m_usageMode = usageMode;
  buf->m_syncDescriptor = syncDescriptor;

  // Permissive usage on the VK side so the buffer can stand in anywhere we
  // choose to bind it on the GL side. CUDA accesses the underlying memory
  // directly and isn't constrained by VkBufferUsage.
  vk::BufferUsageFlags vkUsage =
    vk::BufferUsageFlagBits::eTransferSrc |
    vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer |
    vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;
  buf->m_vkBuffer = vk->allocateExternalBuffer(sizeBytes, vkUsage);

  // GL side: create buffer, import memory.
  GL(glGenBuffers(1, &buf->m_buffer));
  GL(glBindBuffer(GL_ARRAY_BUFFER, buf->m_buffer));

  GL(glCreateMemoryObjectsEXT(1, &buf->m_glMemoryObject));
  int memFdForGL = dup(buf->m_vkBuffer.memoryFd);
  GL(glImportMemoryFdEXT(buf->m_glMemoryObject, buf->m_vkBuffer.allocationSize, GL_HANDLE_TYPE_OPAQUE_FD_EXT, memFdForGL));
  GL(glBufferStorageMemEXT(GL_ARRAY_BUFFER, buf->m_vkBuffer.allocationSize, buf->m_glMemoryObject, /*offset=*/ 0));
  GL(glBindBuffer(GL_ARRAY_BUFFER, 0));

  // CUDA side: import memory + map.
  cudaExternalMemoryHandleDesc memDesc = {};
  memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  memDesc.handle.fd = buf->m_vkBuffer.memoryFd;
  memDesc.size = buf->m_vkBuffer.allocationSize;
  CUDA_CHECK(cudaImportExternalMemory(&buf->m_cudaExtMem, &memDesc));
  buf->m_vkBuffer.memoryFd = -1;

  cudaExternalMemoryBufferDesc bufDesc = {};
  bufDesc.offset = 0;
  bufDesc.size = sizeBytes;
  bufDesc.flags = 0;
  void* cudaPtrRaw = nullptr;
  CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&cudaPtrRaw, buf->m_cudaExtMem, &bufDesc));
  buf->m_cudaPtr = reinterpret_cast<CUdeviceptr>(cudaPtrRaw);
  buf->m_cudaSize = sizeBytes;

  // Register buffer with sync
  syncDescriptor.sync()->addSyncedObject(buf);
  return buf;
}
