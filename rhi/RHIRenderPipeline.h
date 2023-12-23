#pragma once
#include "rhi/RHIObject.h"

enum RHIPrimitiveTopology {
  kPrimitiveTopologyPoints,
  kPrimitiveTopologyLineList,
  kPrimitiveTopologyLineStrip,
  kPrimitiveTopologyTriangleList,
  kPrimitiveTopologyTriangleStrip,
  kPrimitiveTopologyPatches,
};

class RHIShader;
class RHIRenderPipeline : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIRenderPipeline> ptr;
  virtual ~RHIRenderPipeline();

  virtual RHIShader* shader() const = 0;
};

// this should contain everything that we don’t expect to be updated per-frame
// resourceDescriptor and the stream buffers for the vertexDescriptor are draw-time bind
struct RHIRenderPipelineDescriptor {
  RHIRenderPipelineDescriptor(RHIPrimitiveTopology primitiveTopology_ = kPrimitiveTopologyTriangleList) :
    primitiveTopology(primitiveTopology_), patchControlPoints(0), rasterizationEnabled(true), alphaToCoverageEnabled(false), perSampleShadingEnabled(false), primitiveRestartEnabled(false) {}

  // XXX TODO RHIRenderTargetLayout::ptr renderTargetLayout;

  RHIPrimitiveTopology primitiveTopology : 4;
  unsigned int patchControlPoints : 6;
  bool rasterizationEnabled : 1;
  bool alphaToCoverageEnabled : 1;
  bool perSampleShadingEnabled : 1;
  bool primitiveRestartEnabled : 1;

  size_t hash() const;
};

