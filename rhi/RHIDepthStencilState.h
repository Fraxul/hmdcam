#pragma once
#include "rhi/RHIObject.h"
#include <stddef.h>
#include <stdint.h>

// 3 bits. typed to unsigned to allow us to use a narrow bitfield.
enum RHIDepthStencilCompareFunction : unsigned char {
  kCompareNever,
  kCompareLess,
  kCompareLessEqual,
  kCompareEqual,
  kCompareNotEqual,
  kCompareGreaterEqual,
  kCompareGreater,
  kCompareAlways
};

// 3 bits. typed to unsigned to allow us to use a narrow bitfield.
enum RHIStencilOperation : unsigned char {
  kStencilKeep,
  kStencilZero,
  kStencilReplace,
  kStencilIncrementClamp,
  kStencilDecrementClamp,
  kStencilInvert,
  kStencilIncrementWrap,
  kStencilDecrementWrap
};

struct RHIStencilTestStateDescriptor {
  RHIStencilTestStateDescriptor() : failOp(kStencilKeep), depthFailOp(kStencilKeep), passOp(kStencilKeep), compareFunc(kCompareAlways), referenceValue(0) {}

  RHIStencilOperation failOp : 3;
  RHIStencilOperation depthFailOp : 3;
  RHIStencilOperation passOp : 3;
  RHIDepthStencilCompareFunction compareFunc : 3;
  uint8_t referenceValue;
};

struct RHIDepthStencilStateDescriptor {
  RHIDepthStencilStateDescriptor() : depthBoundsMin(0.0f), depthBoundsMax(1.0f), depthTestEnable(false), depthWriteEnable(true), stencilTestEnable(false), depthFunction(kCompareLess), stencilMask(0xff) {}

  RHIStencilTestStateDescriptor stencilFront;
  RHIStencilTestStateDescriptor stencilBack;

  float depthBoundsMin, depthBoundsMax;

  bool depthTestEnable : 1;
  bool depthWriteEnable : 1;
  bool stencilTestEnable : 1;

  RHIDepthStencilCompareFunction depthFunction : 3;

  uint8_t stencilMask;
};

class RHIDepthStencilState : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIDepthStencilState> ptr;
  virtual ~RHIDepthStencilState();

};

