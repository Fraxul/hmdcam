#pragma once
#include "rhi/RHIObject.h"
#include <stddef.h>
#include <stdint.h>

// Forward decl borrowed from <cuda.h>
typedef struct CUgraphicsResource_st *CUgraphicsResource; /**< CUDA graphics interop resource */

enum RHIBufferUsageMode {
  kBufferUsageCPUWriteOnly, // Contents written once by CPU
  kBufferUsageCPUReadback,  // Contents written by GPU, will be read by CPU
  kBufferUsageGPUPrivate    // Contents written and read by GPU only
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

  virtual CUgraphicsResource& cuGraphicsResource() const = 0;
protected:
  RHIBuffer();
  void* m_data;
  size_t m_size;
  RHIBufferUsageMode m_usageMode;
};

