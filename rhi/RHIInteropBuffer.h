#pragma once
// RHIInteropBuffer: mixin interface for buffers that are CUDA-mapped in
// addition to being usable through the regular RHIBuffer API.
//
// The concrete classes RHIInteropBufferGL (rhi/vk/RHIInteropBufferGL) and
// RHIInteropBufferVK (rhi/vk/RHIInteropBufferVK) each inherit from this
// interface *plus* the backend's concrete RHIBuffer type (RHIBufferGL or
// RHIBufferVK). So a `RHIBuffer::ptr` returned by RHI::newInteropBuffer
// can be:
//   - bound through the standard RHI APIs (bindStreamBuffer,
//     drawIndexedPrimitivesIndirect, ...), and
//   - dynamic_cast'd to RHIInteropBuffer* to retrieve the CUDA-side
//     mapping (cudaPointer / cudaSize).
//
// Use a non-owning cached RHIInteropBuffer* alongside the RHIBuffer::ptr
// to avoid repeating the dynamic_cast in hot paths.

#include <cuda.h>
#include <stddef.h>

class RHIInteropBuffer {
public:
  virtual ~RHIInteropBuffer() = default;

  virtual CUdeviceptr cudaPointer() const = 0;
  virtual size_t cudaSize() const = 0;
};
