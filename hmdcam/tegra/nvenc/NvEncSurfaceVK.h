#pragma once
// NvEncSurfaceVK: an RGBA pitch-linear NvBufSurface wrapped as a renderable
// VkImage color attachment.
//
// This is the render-target counterpart to ArgusCameraSurfaceVK. Both import
// an NvBufSurface's DMA-BUF FD into Vulkan; ArgusCameraSurfaceVK imports a
// two-plane NV12 surface for *sampling* (ycbcr), while this imports a
// single-plane RGBA surface for *rendering* (color attachment). The RHI draws
// the remote-debug view directly into this surface; the NvEncSession worker
// then hands the same NvBufSurface to the Tegra VIC (NvBufSurfTransform) for
// RGBA->YUV420 conversion on its way to the H.264 encoder -- no CUDA copy and
// no EGLImage interop, which the GL-era path required.
//
// Import path mirrors the validated ArgusCameraSurfaceVK spike, minus the
// disjoint multi-plane handling:
//   - VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_LINEAR.
//   - Single VkDeviceMemory imported from the NvBuf DMA-BUF FD; the image is
//     bound at the plane-0 offset.
//   - VkImageView with COLOR aspect (no ycbcr conversion).
//   - VK_IMAGE_USAGE_COLOR_ATTACHMENT so beginRenderPass can target it.
//
// We commit to direct render (no copy fallback): create() aborts if the driver
// won't accept the imported linear image as a color attachment. The
// NvBufSurface color format / layout can be retuned here if a different
// VK-renderable + VIC-consumable combination is needed.

#include "rhi/vk/RHISurfaceVK.h"

struct NvBufSurface;

class NvEncSurfaceVK : public RHISurfaceVK {
public:
  typedef boost::intrusive_ptr<NvEncSurfaceVK> ptr;

  // Allocate a width x height RGBA pitch-linear NvBufSurface and import its
  // DMA-BUF into Vulkan as a color-attachment image. Aborts on any failure.
  static NvEncSurfaceVK::ptr create(uint32_t width, uint32_t height);

  virtual ~NvEncSurfaceVK();

  // The underlying NvBufSurface, for handoff to NvBufSurfTransform (VIC).
  NvBufSurface* nvBufSurface() const { return m_surf; }

protected:
  NvEncSurfaceVK() = default;

  NvBufSurface* m_surf = nullptr;

  // VkImage extent.width is padded to NvBuf's per-row pitch (the driver's
  // tight-pack rowPitch on a logical width wouldn't match NvBuf's hardware-
  // aligned pitch). Rendering targets the logical [0, width()) sub-region; VIC
  // reads the same logical width at the surface pitch, so the padding columns
  // are inert. See ArgusCameraSurfaceVK for the same reasoning.
  uint32_t m_extentWidth = 0;
};
