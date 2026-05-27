#pragma once
// ArgusCameraSurfaceVK: wraps an NvBufSurface (NV12 pitch-linear) so it can be
// sampled as a VkImage with ycbcr conversion AND read by CUDA as per-plane
// Y/UV CUtexObjects.
//
// This is the Vulkan-side replacement for the EGLImage path inside
// ArgusCamera. Argus itself still consumes an EGLImage handle (the only buffer
// type its API supports), but the EGLImage is just a lightweight handle over
// the underlying NvBuf; the actual storage is shared. This class imports the
// same NvBuf's DMA-BUF FD into VK and CUDA independently, sidestepping the
// GL-CUDA-via-EGL chain that used to be required for rendering and
// disparity-feed access.
//
// VK import path matches the validated NV12+ycbcr spike:
//   - VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, VK_IMAGE_TILING_LINEAR,
//     VK_IMAGE_CREATE_DISJOINT_BIT.
//   - vkBindImageMemory2 with one VkBindImagePlaneMemoryInfo per plane,
//     both bindings pointing into the single imported VkDeviceMemory at the
//     plane offsets reported by NvBufSurfacePlaneParams.
//   - VkImageView created with the supplied VkSamplerYcbcrConversion attached
//     via VkSamplerYcbcrConversionInfo in pNext. A sampler created with the
//     same conversion is required at descriptor-binding time (typically
//     declared as an immutable sampler on the camera pipeline's layout).
//
// CUDA import: cudaImportExternalMemory on a dup of the same FD, then
// cudaExternalMemoryGetMappedBuffer per plane (at plane offset, with plane
// size). The resulting device pointers + plane pitches feed
// CU_RESOURCE_TYPE_PITCH2D CUtexObjects for Y and UV, preserving the
// existing IArgusCamera::cudaLuma/ChromaTexObject contract that drives
// depth + eyetracking + greyscale-mat consumers.

#include "rhi/vk/RHISurfaceVK.h"
#include <cuda.h>
#include <cuda_runtime.h>

struct NvBufSurface;

class ArgusCameraSurfaceVK : public RHISurfaceVK {
public:
  typedef boost::intrusive_ptr<ArgusCameraSurfaceVK> ptr;

  // Imports `surf` (must be NV12 pitch-linear with 2 planes) per the doc
  // above. The conversion handle is borrowed (not owned) — the caller
  // (ArgusCamera) owns a shared RHISamplerVK that holds the conversion.
  // Takes ownership of `surf`: the NvBufSurface is destroyed in the
  // dtor after VK + CUDA references are released.
  static ArgusCameraSurfaceVK::ptr create(NvBufSurface* surf, vk::SamplerYcbcrConversion conversion);

  virtual ~ArgusCameraSurfaceVK();

  CUtexObject cudaLumaTexObject() const { return m_cudaLumaTexObject; }
  CUtexObject cudaChromaTexObject() const { return m_cudaChromaTexObject; }

  // Per-plane raw access for IArgusCamera::fillCudaMemcpy2DForStreamSource.
  // chromaPlane=false -> luma (full-height, single channel), true -> chroma
  // (half-height, two channels). srcPitch is shared between the planes by
  // NvBuf's convention. WidthInBytes equals the Y plane width either way
  // (chroma has half the width but twice the bytes per sample).
  bool fillCudaMemcpy2D(CUDA_MEMCPY2D& out, bool chromaPlane) const;

  // For gpuMatGreyscale: luma plane width / height / pitch + device ptr.
  uint32_t lumaWidth() const { return m_widthY; }
  uint32_t lumaHeight() const { return m_heightY; }
  size_t lumaPitch() const { return m_pitchY; }
  CUdeviceptr lumaDevicePtr() const { return m_cudaY; }

protected:
  ArgusCameraSurfaceVK() = default;

  NvBufSurface* m_surf = nullptr;

  // VK side: m_image / m_memory / m_imageView are inherited from RHISurfaceVK.
  // We just populate them with externally-imported handles in create().

  // CUDA side.
  cudaExternalMemory_t m_cudaExtMem = nullptr;
  CUdeviceptr m_cudaY = 0;
  CUdeviceptr m_cudaUV = 0;
  size_t m_pitchY = 0;
  size_t m_pitchUV = 0;
  uint32_t m_widthY = 0;
  uint32_t m_heightY = 0;
  uint32_t m_widthUV = 0;
  uint32_t m_heightUV = 0;
  CUtexObject m_cudaLumaTexObject = 0;
  CUtexObject m_cudaChromaTexObject = 0;
};
