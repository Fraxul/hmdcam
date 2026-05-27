#pragma once
// RHISamplerVK: wraps a VkSampler created from an RHISamplerDescriptor.
//
// If the descriptor specifies an RHIYcbcrConversionDescriptor with a non-None
// sourceFormat, the sampler also carries a VkSamplerYcbcrConversion. ycbcr
// samplers must be bound as immutable samplers in the descriptor set layout
// (see RHIRenderPipelineVK), and any VkImageView sampled through them must be
// created with the same ycbcr conversion attached.

#include "rhi/RHISurface.h" // RHISampler + RHISamplerDescriptor
#include "rhi/vk/RHIVulkan.h"

class RHISamplerVK : public RHISampler {
public:
  typedef boost::intrusive_ptr<RHISamplerVK> ptr;

  static RHISamplerVK::ptr create(const RHISamplerDescriptor&);

  virtual ~RHISamplerVK();

  vk::Sampler vkSampler() const { return m_sampler.get(); }

  // VK_NULL_HANDLE when the descriptor's ycbcrConversion is not enabled.
  // Otherwise: the conversion this sampler was created against. The same
  // handle must be referenced from any VkImageView sampled through this
  // sampler (passed via VkSamplerYcbcrConversionInfo in the view's pNext).
  vk::SamplerYcbcrConversion vkSamplerYcbcrConversion() const { return m_ycbcrConversion.get(); }
  bool hasYcbcrConversion() const { return static_cast<bool>(m_ycbcrConversion); }

  const RHIYcbcrConversionDescriptor& ycbcrConversionDescriptor() const { return m_ycbcrConversionDescriptor; }

protected:
  RHISamplerVK() = default;
  vk::UniqueSampler m_sampler;
  vk::UniqueSamplerYcbcrConversion m_ycbcrConversion;
  RHIYcbcrConversionDescriptor m_ycbcrConversionDescriptor;
};
