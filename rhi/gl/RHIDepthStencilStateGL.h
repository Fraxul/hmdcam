#pragma once
#include "rhi/RHIDepthStencilState.h"

class RHIDepthStencilStateGL : public RHIDepthStencilState {
public:
  typedef boost::intrusive_ptr<RHIDepthStencilStateGL> ptr;
  RHIDepthStencilStateGL(const RHIDepthStencilStateDescriptor&);
  virtual ~RHIDepthStencilStateGL();

  const RHIDepthStencilStateDescriptor& descriptor() const { return m_descriptor; }
protected:
  RHIDepthStencilStateDescriptor m_descriptor;
};

