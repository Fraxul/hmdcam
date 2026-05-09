#pragma once
#include <stdint.h>
#include "rhi/RHIRenderTarget.h"

enum ERenderBackend {
  kRenderBackendNone,
  kRenderBackendDRM,
  kRenderBackendWayland,
  kRenderBackendVKDirect,
  kRenderBackendSurfaceless,
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

  // Three-phase initialization. RenderInit() invokes these in order, with
  // shared global setup (CUDA, RHI/GL, RHI/Vulkan) interleaved between
  // phases — see Render.cpp.
  //
  // earlyInit(): backend-specific work that must happen before a GL context
  //   exists. Default empty; backends that need to power on a display, open a
  //   native display connection, or pre-pick a DRM device override here.
  //
  // createGLContext(): create the EGL display and OpenGL context, then
  //   eglMakeCurrent() it. After this returns, GL calls are valid. Backends
  //   that receive a host-created GL context (SDL, embedded uses) can leave
  //   this empty.
  //
  // createPresentation(): build the swapchain / window surface that GL will
  //   render into and that the backend will present from. May use
  //   rhi()->vk() — RHIVulkan is initialized before this is called.
  virtual void earlyInit() {}
  virtual void createGLContext() = 0;
  virtual void createPresentation() = 0;

  virtual uint32_t surfaceWidth() const = 0;
  virtual uint32_t surfaceHeight() const = 0;
  virtual double refreshRateHz() const = 0;

  virtual EGLDisplay eglDisplay() const = 0;
  virtual EGLContext eglContext() const = 0;
  virtual EGLSurface eglSurface() const = 0;
  virtual EGLConfig eglConfig() const = 0;

  virtual RHIRenderTarget::ptr windowRenderTarget() const = 0;

  // Returns the timestamp (in nanoseconds, referenced to currentTimeNs()) of the
  // most recent first-pixel-out scanout event. Returns 0 if not supported by this backend.
  virtual uint64_t lastPresentationTimestamp() const = 0;
};
