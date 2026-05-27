#pragma once
// RHISamplerVK: wraps a VkSampler created from an RHISamplerDescriptor.

#include "rhi/RHISurface.h" // RHISampler + RHISamplerDescriptor
#include "rhi/vk/RHIVulkan.h"

class RHISamplerVK : public RHISampler {
public:
  typedef boost::intrusive_ptr<RHISamplerVK> ptr;

  static RHISamplerVK::ptr create(const RHISamplerDescriptor&);

  virtual ~RHISamplerVK();

  vk::Sampler vkSampler() const { return m_sampler.get(); }

protected:
  RHISamplerVK() = default;
  vk::UniqueSampler m_sampler;
};
