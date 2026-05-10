#pragma once
// RHIInteropSurfaceGL: a GL texture and a CUDA cudaArray backed by the same
// Vulkan-allocated VkImage (optimal tiling).
//
// Tiling note: VK and GL both use optimal/swizzled tiling. NVIDIA Tegra's
// GL_EXT_memory_object implementation does not accept GL_LINEAR_TILING_EXT
// for imported memory, so a CUDA-side pitched-buffer view (cv::cuda::GpuMat)
// is not available. CUDA accesses the surface as a cudaArray with surface
// and texture objects layered on top.
//
// Layout: GL_LAYOUT_GENERAL_EXT for both directions; the first signal on
// either side handles the undefined→general transition implicitly.

#include "rhi/vk/RHIInteropSyncDescriptor.h"
#include "rhi/gl/RHISurfaceGL.h"
#include "rhi/vk/RHIVulkan.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

class RHIInteropSurfaceGL : public RHISurfaceGL {
public:
  typedef boost::intrusive_ptr<RHIInteropSurfaceGL> ptr;

  static RHIInteropSurfaceGL* newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&, const RHIInteropSyncDescriptor&);

  virtual ~RHIInteropSurfaceGL();

  // Stable across the lifetime of this surface.
  cudaArray_t cudaArray() const { return m_cudaArray; }

  // Convenience surface and texture objects bound to the underlying CUarray.
  // The texture object uses point filter, clamp address, element-type reads,
  // non-normalized coords; reset by direct cudaArray() use if you need other
  // semantics.
  cudaSurfaceObject_t cudaSurfaceObject() const { return m_cudaSurfaceObject; }
  cudaTextureObject_t cudaTextureObject() const { return m_cudaTextureObject; }

  // Convenience: 2D async copy from src GpuMat into this surface's CUarray
  // on the given stream. Drop-in replacement for RHICUDA::copyGpuMatToSurface()
  // that doesn't pay for a per-frame cuGraphicsMap/Unmap. Caller is
  // responsible for calling signalCUDADone(stream) afterwards.
  void copyFromGpuMatAsync(const cv::cuda::GpuMat& src, cudaStream_t stream);

  const RHIInteropSyncDescriptor& syncDescriptor() const { return m_syncDescriptor; };

private:
  friend class RHIInteropSync;
  RHIInteropSurfaceGL();

  RHIVulkan::ExternalImage m_vkImage;
  GLuint m_glMemoryObject = 0;

  cudaExternalMemory_t m_cudaExtMem = nullptr;
  cudaMipmappedArray_t m_cudaMipmappedArray = nullptr;
  cudaArray_t m_cudaArray = nullptr; // level 0 of m_cudaMipmappedArray
  cudaSurfaceObject_t m_cudaSurfaceObject = 0;
  cudaTextureObject_t m_cudaTextureObject = 0;

  RHIInteropSyncDescriptor m_syncDescriptor;
};
