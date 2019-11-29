#pragma once
#include "rhi/gl/RHIWindowRenderTargetGL.h"

class RHIEGLSurfaceRenderTargetGL : public RHIWindowRenderTargetGL {
public:
  typedef boost::intrusive_ptr<RHIEGLSurfaceRenderTargetGL> ptr;

  RHIEGLSurfaceRenderTargetGL(EGLDisplay, EGLSurface);
  virtual ~RHIEGLSurfaceRenderTargetGL();

  virtual void platformSwapBuffers();

  EGLDisplay display() const { return m_display; }
  EGLSurface surface() const { return m_surface; }

protected:
  EGLDisplay m_display;
  EGLSurface m_surface;
};

