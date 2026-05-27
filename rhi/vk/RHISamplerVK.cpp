#include "rhi/vk/RHISamplerVK.h"
#include "rhi/RHI.h"

namespace {

vk::SamplerAddressMode vkWrap(RHISamplerWrapMode w) {
  switch (w) {
    case kWrapClamp: return vk::SamplerAddressMode::eClampToEdge;
    case kWrapRepeat: return vk::SamplerAddressMode::eRepeat;
  }
  return vk::SamplerAddressMode::eClampToEdge;
}

} // namespace

RHISamplerVK::~RHISamplerVK() {}

/*static*/ RHISamplerVK::ptr RHISamplerVK::create(const RHISamplerDescriptor& d) {
  vk::SamplerCreateInfo ci{};
  switch (d.filter) {
    case kFilterNearest:
      ci.magFilter = vk::Filter::eNearest;
      ci.minFilter = vk::Filter::eNearest;
      ci.mipmapMode = vk::SamplerMipmapMode::eNearest;
      break;
    case kFilterLinear:
      ci.magFilter = vk::Filter::eLinear;
      ci.minFilter = vk::Filter::eLinear;
      ci.mipmapMode = vk::SamplerMipmapMode::eNearest;
      break;
    case kFilterMipLinear:
      ci.magFilter = vk::Filter::eLinear;
      ci.minFilter = vk::Filter::eLinear;
      ci.mipmapMode = vk::SamplerMipmapMode::eLinear;
      break;
  }
  ci.addressModeU = vkWrap(d.wrapModeU);
  ci.addressModeV = vkWrap(d.wrapModeV);
  ci.addressModeW = vk::SamplerAddressMode::eRepeat;
  if (d.maxAnisotropy > 1) {
    ci.anisotropyEnable = VK_TRUE;
    ci.maxAnisotropy = static_cast<float>(d.maxAnisotropy);
  }
  ci.minLod = 0.0f;
  ci.maxLod = VK_LOD_CLAMP_NONE;
  ci.borderColor = vk::BorderColor::eFloatTransparentBlack;

  RHISamplerVK::ptr s(new RHISamplerVK());
  s->m_sampler = rhi()->vk()->device().createSamplerUnique(ci);
  return s;
}
