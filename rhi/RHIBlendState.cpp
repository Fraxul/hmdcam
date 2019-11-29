#include "rhi/RHIBlendState.h"
#include <boost/functional/hash.hpp>

size_t RHIBlendStateDescriptorElement::hash() const {
  size_t h = boost::hash_value(blendEnabled);
  boost::hash_combine(h, boost::hash_value(colorSource));
  boost::hash_combine(h, boost::hash_value(colorDest));
  boost::hash_combine(h, boost::hash_value(alphaSource));
  boost::hash_combine(h, boost::hash_value(alphaDest));
  boost::hash_combine(h, boost::hash_value(colorFunc));
  boost::hash_combine(h, boost::hash_value(alphaFunc));
  return h;
}

size_t RHIBlendStateDescriptor::hash() const {
  size_t h = boost::hash_value(targetBlendStates.size());
  for (size_t i = 0; i < targetBlendStates.size(); ++i) {
    boost::hash_combine(h, targetBlendStates[i].hash());
  }

  for (size_t i = 0; i < 4; ++i) {
    boost::hash_combine(h, boost::hash_value(constantColor[i]));
  }
  return h;
}

RHIBlendState::~RHIBlendState() {

}

