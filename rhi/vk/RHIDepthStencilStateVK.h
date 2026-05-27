#pragma once
// RHIDepthStencilStateVK: same story as RHIBlendStateVK — depth/stencil is
// dynamic state with shader_object + extended_dynamic_state, so this is
// just a descriptor wrapper.

#include "rhi/RHIDepthStencilState.h"

class RHIDepthStencilStateVK : public RHIDepthStencilState {
public:
  typedef boost::intrusive_ptr<RHIDepthStencilStateVK> ptr;
  explicit RHIDepthStencilStateVK(const RHIDepthStencilStateDescriptor& d) :
    m_descriptor(d) {}
  virtual ~RHIDepthStencilStateVK() {}

  const RHIDepthStencilStateDescriptor& descriptor() const { return m_descriptor; }

private:
  RHIDepthStencilStateDescriptor m_descriptor;
};
