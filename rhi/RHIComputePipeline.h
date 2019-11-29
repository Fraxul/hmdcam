#pragma once
#include "rhi/RHIObject.h"

class RHIShader;
class RHIComputePipeline : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIComputePipeline> ptr;
  virtual ~RHIComputePipeline();

  virtual RHIShader* shader() const = 0;
};

