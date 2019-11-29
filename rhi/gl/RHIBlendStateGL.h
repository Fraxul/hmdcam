#pragma once
#include "rhi/RHIBlendState.h"

class RHIBlendStateGL : public RHIBlendState {
public:
  typedef boost::intrusive_ptr<RHIBlendStateGL> ptr;
  RHIBlendStateGL(const RHIBlendStateDescriptor&);
  virtual ~RHIBlendStateGL();

  const RHIBlendStateDescriptor& descriptor() const { return m_descriptor; }
protected:
  RHIBlendStateDescriptor m_descriptor;
};
