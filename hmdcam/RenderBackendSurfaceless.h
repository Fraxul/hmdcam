#pragma once
#include "RenderBackend.h"
#include <epoxy/egl.h>
#include "rhi/egl/RHIEGLSurfaceRenderTargetGL.h"

class RenderBackendSurfaceless : public RenderBackend {
public:
  RenderBackendSurfaceless();
  virtual ~RenderBackendSurfaceless();

  virtual void createGLContext() override;
  virtual void createPresentation() override;

  virtual uint32_t surfaceWidth() const override { return m_surfaceWidth; }
  virtual uint32_t surfaceHeight() const override { return m_surfaceHeight; }
  virtual double refreshRateHz() const override { return m_refreshRateHz; }

  virtual EGLDisplay eglDisplay() const override { return m_eglDisplay; }
  virtual EGLContext eglContext() const override { return m_eglContext; }
  virtual EGLSurface eglSurface() const override { return m_eglSurface; }
  virtual EGLConfig eglConfig() const override { return m_eglConfig; }

  virtual RHIRenderTarget::ptr windowRenderTarget() const override { return m_windowRenderTarget; }

  virtual uint64_t lastPresentationTimestamp() const override { return 0; }

private:
  uint32_t m_surfaceWidth = 0;
  uint32_t m_surfaceHeight = 0;
  double m_refreshRateHz = 0;

  // Internal state
  EGLDisplay m_eglDisplay = EGL_NO_DISPLAY;
  EGLContext m_eglContext = NULL;

  EGLConfig m_eglConfig = NULL;
  EGLSurface m_eglSurface = NULL;

  RHIEGLSurfaceRenderTargetGL::ptr m_windowRenderTarget;

  // noncopyable
  RenderBackendSurfaceless(const RenderBackendSurfaceless&);
  RenderBackendSurfaceless& operator=(const RenderBackendSurfaceless&);
};
