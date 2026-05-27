#pragma once
// RHIInteropSurfaceVK: a VkImage + CUDA cudaArray backed by the same
// Vulkan-allocated optimal-tiled image. The VK side is a regular
// RHISurfaceVK so existing draw paths (loadTexture / loadImage,
// render-target binding) work unchanged. The CUDA side gets a
// cudaMipmappedArray import + cudaArray for level 0 + ready-made
// surface/texture objects.
//
// Replaces the GL-counterpart RHIInteropSurfaceGL for the VK RHI backend.

#include "rhi/RHIInteropSurface.h"
#include "rhi/vk/RHISurfaceVK.h"
#include "rhi/vk/RHIInteropSyncDescriptor.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

class RHIInteropSurfaceVK : public RHISurfaceVK, public RHIInteropSurface {
public:
  typedef boost::intrusive_ptr<RHIInteropSurfaceVK> ptr;

  static RHIInteropSurfaceVK* newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&, const RHIInteropSyncDescriptor&);

  virtual ~RHIInteropSurfaceVK();

  cudaArray_t cudaArray() const override { return m_cudaArray; }
  cudaSurfaceObject_t cudaSurfaceObject() const override { return m_cudaSurfaceObject; }
  cudaTextureObject_t cudaTextureObject() const override { return m_cudaTextureObject; }

  void copyFromGpuMatAsync(const cv::cuda::GpuMat& src, cudaStream_t stream) override;

  const RHIInteropSyncDescriptor& syncDescriptor() const { return m_syncDescriptor; }

private:
  friend class RHIInteropSync;
  RHIInteropSurfaceVK();

  cudaExternalMemory_t m_cudaExtMem = nullptr;
  cudaMipmappedArray_t m_cudaMipmappedArray = nullptr;
  cudaArray_t m_cudaArray = nullptr; // level 0 of m_cudaMipmappedArray
  cudaSurfaceObject_t m_cudaSurfaceObject = 0;
  cudaTextureObject_t m_cudaTextureObject = 0;

  RHIInteropSyncDescriptor m_syncDescriptor;
};
