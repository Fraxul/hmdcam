#include "RDMABuffer.h"
#include <cuda.h>
#include "rhi/cuda/CudaUtil.h"

RDMABuffer::RDMABuffer(RDMAContext* ctx, const std::string& key, char* bufferData, size_t bufferSize, RDMABufferUsage usage)
  : m_context(ctx), m_key(key), m_bufferData(bufferData), m_bufferSize(bufferSize), m_mr(NULL), m_usage(usage) {

}

RDMABuffer::~RDMABuffer() {
  if (m_mr) {
    ibv_dereg_mr(m_mr);
  }
}

RDMAManagedBuffer::RDMAManagedBuffer(RDMAContext* ctx, const std::string& key, size_t bufferSize, RDMABufferUsage usage)
  : RDMABuffer(ctx, key, NULL, bufferSize, usage) {

  void* data = NULL;
  posix_memalign(&data, sysconf(_SC_PAGESIZE), bufferSize);
  assert(data);
  memset(data, 0, bufferSize);

  m_bufferData = reinterpret_cast<char*>(data);
}

RDMAManagedBuffer::~RDMAManagedBuffer() {
  free(m_bufferData); // allocated with posix_memalign
}

RDMACUDAManagedBuffer::RDMACUDAManagedBuffer(RDMAContext* ctx, const std::string& key, size_t bufferSize, RDMABufferUsage usage)
  : RDMABuffer(ctx, key, NULL, bufferSize, usage) {

  void* data = NULL;
  CUDA_CHECK(cuMemAllocHost(&data, bufferSize));
  memset(data, 0, bufferSize);

  m_bufferData = reinterpret_cast<char*>(data);
}

RDMACUDAManagedBuffer::~RDMACUDAManagedBuffer() {
  cuMemFreeHost(m_bufferData);
}

