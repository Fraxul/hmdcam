#pragma once
#include "rhi/RHIConstants.h"
#include "rhi/RHIObject.h"
#include <glm/glm.hpp>
#include <initializer_list>
#include <boost/container/static_vector.hpp>

enum RHIBlendWeight : unsigned char {
  kBlendZero,
  kBlendOne,
  kBlendSourceColor,
  kBlendOneMinusSourceColor,
  kBlendDestColor,
  kBlendOneMinusDestColor,
  kBlendSourceAlpha,
  kBlendOneMinusSourceAlpha,
  kBlendDestAlpha,
  kBlendOneMinusDestAlpha,
  kBlendConstantColor,
  kBlendOneMinusConstantColor,
  kBlendConstantAlpha,
  kBlendOneMinusConstantAlpha,
  kBlendSourceAlphaSaturate,
  kBlendSource1Color,
  kBlendOneMinusSource1Color,
  kBlendSource1Alpha,
  kBlendOneMinusSource1Alpha
};

enum RHIBlendFunc : unsigned char {
  kBlendFuncAdd,
  kBlendFuncSubtract,
  kBlendFuncReverseSubtract,
  kBlendFuncMin,
  kBlendFuncMax
};

struct RHIBlendStateDescriptorElement {
  RHIBlendStateDescriptorElement() : blendEnabled(false), colorSource(kBlendOne), colorDest(kBlendZero), alphaSource(kBlendOne), alphaDest(kBlendZero), colorFunc(kBlendFuncAdd), alphaFunc(kBlendFuncAdd) {}
  RHIBlendStateDescriptorElement(RHIBlendWeight source, RHIBlendWeight dest, RHIBlendFunc func = kBlendFuncAdd) : blendEnabled(true), colorSource(source), colorDest(dest), alphaSource(source), alphaDest(dest), colorFunc(func), alphaFunc(func) {}
  RHIBlendStateDescriptorElement(RHIBlendWeight colorSource_, RHIBlendWeight colorDest_, RHIBlendWeight alphaSource_, RHIBlendWeight alphaDest_, RHIBlendFunc func = kBlendFuncAdd) : blendEnabled(true), colorSource(colorSource_), colorDest(colorDest_), alphaSource(alphaSource_), alphaDest(alphaDest_), colorFunc(func), alphaFunc(func) {}
  bool blendEnabled;
  RHIBlendWeight colorSource, colorDest;
  RHIBlendWeight alphaSource, alphaDest;
  RHIBlendFunc colorFunc, alphaFunc;

  size_t hash() const;
};

struct RHIBlendStateDescriptor {
  RHIBlendStateDescriptor(const RHIBlendStateDescriptorElement& el, const glm::vec4& constantColor_ = glm::vec4(0.0f)) : constantColor(constantColor_) { targetBlendStates.push_back(el); }
  RHIBlendStateDescriptor(const std::initializer_list<RHIBlendStateDescriptorElement>& els, const glm::vec4& constantColor_ = glm::vec4(0.0f)) : targetBlendStates(els.begin(), els.end()), constantColor(constantColor_) {}
  RHIBlendStateDescriptor() : constantColor(glm::vec4(0.0f)) {}

  boost::container::static_vector<RHIBlendStateDescriptorElement, kRHIMaxColorRenderTargets> targetBlendStates;
  glm::vec4 constantColor;

  size_t hash() const;
};

class RHIBlendState : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIBlendState> ptr;
  virtual ~RHIBlendState();

};

