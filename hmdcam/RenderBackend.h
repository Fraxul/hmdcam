#pragma once
#include <epoxy/egl.h>

class RenderBackend {
public:
  virtual ~RenderBackend() {}
  virtual uint32_t surfaceWidth() const = 0;
  virtual uint32_t surfaceHeight() const = 0;
  virtual double refreshRateHz() const = 0;

  virtual EGLDisplay eglDisplay() const = 0;
  virtual EGLContext eglContext() const = 0;
  virtual EGLSurface eglSurface() const = 0;
  virtual EGLConfig eglConfig() const = 0;
};

