#pragma once
#include "rhi/RHISurface.h"
#include "rhi/vk/RHIVulkan.h"

class RHISurfaceVK : public RHISurface {
public:
  typedef boost::intrusive_ptr<RHISurfaceVK> ptr;

  static RHISurfaceVK::ptr newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&);

  virtual ~RHISurfaceVK();

  virtual bool isValidRenderTarget() const override { return true; }
  virtual bool isValidTextureSource() const override { return true; }
  virtual RHISurfaceFormat format() const override { return m_rhiFormat; }
  virtual uint32_t width() const override { return m_width; }
  virtual uint32_t height() const override { return m_height; }
  virtual uint32_t depth() const override { return m_depth; }
  virtual uint32_t layers() const override { return m_layers; }
  virtual uint32_t samples() const override { return m_samples; }
  virtual uint32_t mipLevels() const override { return m_levels; }
  virtual bool isArray() const override { return m_isArrayTexture; }

  vk::Image vkImage() const { return m_image.get(); }
  vk::ImageView vkImageView() const { return m_imageView.get(); }
  vk::Format vkFormat() const { return m_vkFormat; }

  // Upload `data` into mip level 0, layer 0 of this image. Data layout must
  // match the surface format (tightly packed).
  void loadFromCPU(const void* data, size_t bytes);

protected:
  RHISurfaceVK() = default;

  vk::UniqueImage m_image;
  vk::UniqueDeviceMemory m_memory;
  vk::UniqueImageView m_imageView;
  vk::Format m_vkFormat{vk::Format::eUndefined};

  RHISurfaceFormat m_rhiFormat{kSurfaceFormat_Invalid};
  uint32_t m_width = 0;
  uint32_t m_height = 0;
  uint32_t m_depth = 1;
  uint32_t m_layers = 1;
  uint32_t m_samples = 1;
  uint32_t m_levels = 1;
  bool m_isArrayTexture = false;
};

// Format mapping helper. Aborts on unsupported format.
vk::Format rhiSurfaceFormatToVK(RHISurfaceFormat);
