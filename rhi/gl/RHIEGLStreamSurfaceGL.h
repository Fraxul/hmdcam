#pragma once
#include "rhi/gl/RHISurfaceGL.h"

class RHIEGLStreamSurfaceGL : public RHISurfaceGL {
public:
  typedef boost::intrusive_ptr<RHIEGLStreamSurfaceGL> ptr;
  virtual ~RHIEGLStreamSurfaceGL();

  RHIEGLStreamSurfaceGL(uint32_t width_, uint32_t height_, RHISurfaceFormat format_);

};

