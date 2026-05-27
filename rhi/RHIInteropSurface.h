#pragma once
// RHIInteropSurface: mixin interface for surfaces (textures) that are
// CUDA-mapped in addition to being usable through the regular RHISurface
// API.
//
// Same pattern as RHIInteropBuffer: concrete classes RHIInteropSurfaceGL
// (rhi/vk/RHIInteropSurfaceGL) and RHIInteropSurfaceVK (rhi/vk/
// RHIInteropSurfaceVK) each inherit from this interface *plus* the
// backend's concrete RHISurface type (RHISurfaceGL or RHISurfaceVK). So a
// `RHISurface::ptr` returned by RHI::newInteropSurface can be:
//   - bound through the standard RHI loadTexture/loadImage APIs, and
//   - dynamic_cast'd to RHIInteropSurface* to retrieve the CUDA-side
//     mapping (cudaArray, cudaSurfaceObject, cudaTextureObject).
//
// Use a cached non-owning RHIInteropSurface* alongside the RHISurface::ptr
// to avoid repeating the dynamic_cast in hot paths.

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

class RHIInteropSurface {
public:
  virtual ~RHIInteropSurface() = default;

  // Stable across the surface's lifetime.
  virtual cudaArray_t cudaArray() const = 0;

  // Convenience surface/texture objects bound to the underlying array.
  // The texture object uses point filter, clamp address, element-type
  // reads, non-normalized coords; re-create via cudaArray() if you need
  // different semantics.
  virtual cudaSurfaceObject_t cudaSurfaceObject() const = 0;
  virtual cudaTextureObject_t cudaTextureObject() const = 0;

  // 2D async memcpy from src GpuMat into this surface's underlying
  // cudaArray on the given stream. The copy is clipped to the smaller
  // of source/destination width and height. Caller is responsible for
  // any signal/wait via the RHIInteropSync attached at construction.
  virtual void copyFromGpuMatAsync(const cv::cuda::GpuMat& src, cudaStream_t stream) = 0;
};
