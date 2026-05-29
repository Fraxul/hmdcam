#pragma once
#include "rhi/RHIObject.h"
#include <stddef.h>
#include <stdint.h>

// Forward decl borrowed from <cuda.h>
typedef unsigned long long CUdeviceptr_v2;
typedef CUdeviceptr_v2 CUdeviceptr; /**< CUDA device pointer */

enum RHIBufferUsageMode {
  kBufferUsageCPUWriteOnly, // Contents written once by CPU
  kBufferUsageCPUReadback, // Contents written by GPU, will be read by CPU
  kBufferUsageGPUPrivate // Contents written and read by GPU only
};

enum RHIBufferMapMode {
  kBufferMapReadOnly,
  kBufferMapReadWrite,
  kBufferMapWriteOnly
};

class RHIBuffer : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIBuffer> ptr;
  virtual ~RHIBuffer();

  virtual void map(RHIBufferMapMode mapMode) = 0;
  virtual void unmap() = 0;

  void* data() const { return m_data; }
  size_t size() const { return m_size; }
  RHIBufferUsageMode usageMode() const { return m_usageMode; }

  bool isInteropBuffer() const { return m_isInteropBuffer; }
  CUdeviceptr cudaPointer() const {
    assert(m_isInteropBuffer);
    return m_cudaPointer;
  }

protected:
  RHIBuffer() = default;
  void* m_data = nullptr;
  size_t m_size = 0;
  RHIBufferUsageMode m_usageMode = kBufferUsageGPUPrivate;
  bool m_isInteropBuffer = false;
  CUdeviceptr m_cudaPointer = 0;
};
