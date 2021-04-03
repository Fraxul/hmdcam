#include "rhi/RHIRenderPipeline.h"
#include <boost/functional/hash.hpp>

RHIRenderPipeline::~RHIRenderPipeline() {

}

size_t RHIRenderPipelineDescriptor::hash() const {
  size_t h = boost::hash_value(primitiveTopology);
  boost::hash_combine(h, boost::hash_value((primitiveTopology == kPrimitiveTopologyPatches) ? patchControlPoints : 0));
  boost::hash_combine(h, boost::hash_value(rasterizationEnabled));
  boost::hash_combine(h, boost::hash_value(alphaToCoverageEnabled));
  boost::hash_combine(h, boost::hash_value(perSampleShadingEnabled));
  boost::hash_combine(h, boost::hash_value(primitiveRestartEnabled));
  return h;
}

