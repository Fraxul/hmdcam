#pragma once
#include "rhi/gl/RHISurfaceGL.h"

class RHIEGLImageSurfaceGL : public RHISurfaceGL {
public:
  typedef boost::intrusive_ptr<RHIEGLImageSurfaceGL> ptr;
  virtual ~RHIEGLImageSurfaceGL();

  RHIEGLImageSurfaceGL(uint32_t width_, uint32_t height_, RHISurfaceFormat format_);

};


