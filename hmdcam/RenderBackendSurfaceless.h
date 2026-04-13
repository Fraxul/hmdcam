#pragma once
#include "RenderBackend.h"
#include <epoxy/egl.h>
#include "rhi/egl/RHIEGLSurfaceRenderTargetGL.h"

class RenderBackendSurfaceless : public RenderBackend {
public:
  RenderBackendSurfaceless();
  virtual ~RenderBackendSurfaceless();

  virtual void init();

  virtual uint32_t surfaceWidth() const { return m_surfaceWidth; }
  virtual uint32_t surfaceHeight() const { return m_surfaceHeight; }
  virtual double refreshRateHz() const { return m_refreshRateHz; }

  virtual EGLDisplay eglDisplay() const { return m_eglDisplay; }
  virtual EGLContext eglContext() const { return m_eglContext; }
  virtual EGLSurface eglSurface() const { return m_eglSurface; }
  virtual EGLConfig eglConfig() const { return m_eglConfig; }

  virtual RHIRenderTarget::ptr windowRenderTarget() const { return m_windowRenderTarget; }

  virtual uint64_t lastPresentationTimestamp() const { return 0; }

private:

  uint32_t m_surfaceWidth = 0;
  uint32_t m_surfaceHeight = 0;
  double m_refreshRateHz = 0;

  // Internal state
  EGLDeviceEXT m_eglDevice = NULL;
  EGLDisplay m_eglDisplay = EGL_NO_DISPLAY;
  EGLContext m_eglContext = NULL;

  EGLConfig m_eglConfig = NULL;
  EGLSurface m_eglSurface = NULL;

  RHIEGLSurfaceRenderTargetGL::ptr m_windowRenderTarget;

  // noncopyable
  RenderBackendSurfaceless(const RenderBackendSurfaceless&);
  RenderBackendSurfaceless& operator=(const RenderBackendSurfaceless&);
};

