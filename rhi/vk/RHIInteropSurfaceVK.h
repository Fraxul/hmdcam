#pragma once
// RHIInteropSurfaceVK: a VkImage + CUDA cudaArray backed by the same
// Vulkan-allocated optimal-tiled image. The VK side is a regular
// RHISurfaceVK so existing draw paths (loadTexture / loadImage,
// render-target binding) work unchanged. The CUDA side gets a
// cudaMipmappedArray import + cudaArray for level 0

#include "rhi/vk/RHISurfaceVK.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

class RHIInteropSurfaceVK : public RHISurfaceVK {
public:
  typedef boost::intrusive_ptr<RHIInteropSurfaceVK> ptr;

  static RHIInteropSurfaceVK* newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&);

  virtual ~RHIInteropSurfaceVK();

  virtual bool isInteropSurface() const override;
  virtual cudaArray_t cudaArray() const override;

private:
  RHIInteropSurfaceVK();

  cudaExternalMemory_t m_cudaExtMem = nullptr;
  cudaMipmappedArray_t m_cudaMipmappedArray = nullptr;
  cudaArray_t m_cudaArray = nullptr; // level 0 of m_cudaMipmappedArray
};
