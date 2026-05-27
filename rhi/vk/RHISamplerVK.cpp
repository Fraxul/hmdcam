#include "rhi/vk/RHISamplerVK.h"
#include "rhi/RHI.h"
#include <cstdio>

namespace {

vk::SamplerAddressMode vkWrap(RHISamplerWrapMode w) {
  switch (w) {
    case kWrapClamp: return vk::SamplerAddressMode::eClampToEdge;
    case kWrapRepeat: return vk::SamplerAddressMode::eRepeat;
  }
  return vk::SamplerAddressMode::eClampToEdge;
}

vk::Format vkFormatForYcbcrSource(RHIYcbcrConversionDescriptor::SourceFormat f) {
  switch (f) {
    case RHIYcbcrConversionDescriptor::kSourceFormatNV12: return vk::Format::eG8B8R82Plane420Unorm;
    case RHIYcbcrConversionDescriptor::kSourceFormatNone: break;
  }
  fprintf(stderr, "RHISamplerVK: unsupported ycbcr source format %d\n", static_cast<int>(f));
  abort();
}

vk::SamplerYcbcrModelConversion vkYcbcrModel(RHIYcbcrModel m) {
  switch (m) {
    case kYcbcrModelRGBIdentity: return vk::SamplerYcbcrModelConversion::eRgbIdentity;
    case kYcbcrModelYcbcrIdentity: return vk::SamplerYcbcrModelConversion::eYcbcrIdentity;
    case kYcbcrModelYcbcr709: return vk::SamplerYcbcrModelConversion::eYcbcr709;
    case kYcbcrModelYcbcr601: return vk::SamplerYcbcrModelConversion::eYcbcr601;
    case kYcbcrModelYcbcr2020: return vk::SamplerYcbcrModelConversion::eYcbcr2020;
  }
  return vk::SamplerYcbcrModelConversion::eYcbcr709;
}

vk::SamplerYcbcrRange vkYcbcrRange(RHIYcbcrRange r) {
  return (r == kYcbcrRangeFull)
    ? vk::SamplerYcbcrRange::eItuFull
    : vk::SamplerYcbcrRange::eItuNarrow;
}

vk::ChromaLocation vkChromaLocation(RHIChromaLocation c) {
  return (c == kChromaLocationCositedEven)
    ? vk::ChromaLocation::eCositedEven
    : vk::ChromaLocation::eMidpoint;
}

vk::Filter vkChromaFilter(RHISamplerFilterMode f) {
  return (f == kFilterNearest) ? vk::Filter::eNearest : vk::Filter::eLinear;
}

} // namespace

RHISamplerVK::~RHISamplerVK() {}

/*static*/ RHISamplerVK::ptr RHISamplerVK::create(const RHISamplerDescriptor& d) {
  vk::Device device = rhi()->vk()->device();

  // If the descriptor requests ycbcr conversion, create the conversion
  // object first; the sampler and any consuming image view both reference it.
  vk::UniqueSamplerYcbcrConversion ycbcrConv;
  vk::SamplerYcbcrConversionInfo ycbcrSamplerLink;
  if (d.ycbcrConversion.isEnabled()) {
    vk::SamplerYcbcrConversionCreateInfo cci{};
    cci.format = vkFormatForYcbcrSource(d.ycbcrConversion.sourceFormat);
    cci.ycbcrModel = vkYcbcrModel(d.ycbcrConversion.model);
    cci.ycbcrRange = vkYcbcrRange(d.ycbcrConversion.range);
    cci.components = vk::ComponentMapping{};
    cci.xChromaOffset = vkChromaLocation(d.ycbcrConversion.xChromaOffset);
    cci.yChromaOffset = vkChromaLocation(d.ycbcrConversion.yChromaOffset);
    cci.chromaFilter = vkChromaFilter(d.ycbcrConversion.chromaFilter);
    cci.forceExplicitReconstruction = VK_FALSE;
    ycbcrConv = device.createSamplerYcbcrConversionUnique(cci);
    ycbcrSamplerLink.conversion = ycbcrConv.get();
  }

  vk::SamplerCreateInfo ci{};
  if (ycbcrConv) {
    ci.pNext = &ycbcrSamplerLink;
  }
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
  s->m_ycbcrConversion = std::move(ycbcrConv);
  s->m_ycbcrConversionDescriptor = d.ycbcrConversion;
  s->m_sampler = device.createSamplerUnique(ci);
  return s;
}
