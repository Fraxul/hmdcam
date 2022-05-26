#pragma once
#include <stdint.h>
#include "rhi/RHIRenderTarget.h"

enum ERenderBackend {
  kRenderBackendNone,
  kRenderBackendDRM,
  kRenderBackendWayland,
  kRenderBackendVKDirect,
};

ERenderBackend renderBackendStringToEnum(const char*);

typedef void* EGLDisplay;
typedef void* EGLContext;
typedef void* EGLSurface;
typedef void* EGLConfig;

class RenderBackend {
public:
  static RenderBackend* create(ERenderBackend rb);
  virtual ~RenderBackend() {}

  virtual void init() = 0;

  virtual uint32_t surfaceWidth() const = 0;
  virtual uint32_t surfaceHeight() const = 0;
  virtual double refreshRateHz() const = 0;

  virtual EGLDisplay eglDisplay() const = 0;
  virtual EGLContext eglContext() const = 0;
  virtual EGLSurface eglSurface() const = 0;
  virtual EGLConfig eglConfig() const = 0;

  virtual RHIRenderTarget::ptr windowRenderTarget() const = 0;
};

