#pragma once
// RHIBlendStateVK: with VK_EXT_shader_object + extended_dynamic_state3,
// blend state is dynamic — applied at bind time via vkCmdSetXxx calls
// rather than baked into a pipeline. So this class is just a descriptor
// wrapper.

#include "rhi/RHIBlendState.h"

class RHIBlendStateVK : public RHIBlendState {
public:
  typedef boost::intrusive_ptr<RHIBlendStateVK> ptr;
  explicit RHIBlendStateVK(const RHIBlendStateDescriptor& d) :
    m_descriptor(d) {}
  virtual ~RHIBlendStateVK() {}

  const RHIBlendStateDescriptor& descriptor() const { return m_descriptor; }

private:
  RHIBlendStateDescriptor m_descriptor;
};
