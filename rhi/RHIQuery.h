#pragma once
#include "rhi/RHIObject.h"

enum RHIOcclusionQueryMode : unsigned char {
  kOcclusionQueryModeSampleCount,
  kOcclusionQueryModeAnySamplesPassed,
};

class RHIOcclusionQuery : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIOcclusionQuery> ptr;
  virtual ~RHIOcclusionQuery();

protected:
  RHIOcclusionQuery();
};

class RHITimerQuery : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHITimerQuery> ptr;
  virtual ~RHITimerQuery();

protected:
  RHITimerQuery();
};

