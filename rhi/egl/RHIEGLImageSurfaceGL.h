#pragma once
#include "rhi/gl/RHISurfaceGL.h"

class RHIEGLImageSurfaceGL : public RHISurfaceGL {
public:
  typedef boost::intrusive_ptr<RHIEGLImageSurfaceGL> ptr;
  virtual ~RHIEGLImageSurfaceGL();

  static RHIEGLImageSurfaceGL::ptr newTextureExternalOES(EGLImage img, uint32_t width, uint32_t height, RHISurfaceFormat reportedRHIFormat = kSurfaceFormat_RGBA8);

  EGLImage eglImage() const { return m_eglImage; }
protected:
  RHIEGLImageSurfaceGL(EGLImage img, GLenum target);

  void internalQueryTextureInfo();

  EGLImage m_eglImage;

};


