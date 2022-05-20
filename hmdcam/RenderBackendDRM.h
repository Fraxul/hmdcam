#pragma once
#include "RenderBackend.h"
#include <epoxy/egl.h>
#include <xf86drmMode.h>

class RenderBackendDRM : public RenderBackend {
public:
  RenderBackendDRM();
  virtual ~RenderBackendDRM();

  virtual uint32_t surfaceWidth() const { return m_surfaceWidth; }
  virtual uint32_t surfaceHeight() const { return m_surfaceHeight; }
  virtual uint32_t refreshRateHz() const { return m_refreshRateHz; } // approximate/rounded value!

  virtual EGLDisplay eglDisplay() const { return m_eglDisplay; }
  virtual EGLContext eglContext() const { return m_eglContext; }
  virtual EGLSurface eglSurface() const { return m_eglSurface; }
  virtual EGLConfig eglConfig() const { return m_eglConfig; }

private:

  uint32_t m_surfaceWidth = 0;
  uint32_t m_surfaceHeight = 0;
  uint32_t m_refreshRateHz = 0;

  // Internal state
  int m_drmFd = -1;
  EGLDeviceEXT m_eglDevice = NULL;
  EGLDisplay m_eglDisplay = EGL_NO_DISPLAY;
  EGLContext m_eglContext = NULL;

  EGLConfig m_eglConfig = NULL;
  EGLSurface m_eglSurface = NULL;

  EGLOutputLayerEXT m_eglOutputLayer = NULL;
  EGLStreamKHR m_eglStream = NULL;

  drmModeRes* m_drmResources = nullptr;
  drmModeConnector* m_drmConnector = nullptr;
  int m_drmModeIdx = -1;
  drmModeModeInfo* m_drmModeInfo;

  drmModeEncoder* m_drmEncoder = nullptr;
  drmModeCrtc* m_drmCrtc = nullptr;

  struct drm_mode_create_dumb m_drmFb;
  uint32_t m_drmFbBufferId = 0;

  // noncopyable
  RenderBackendDRM(const RenderBackendDRM&);
  RenderBackendDRM& operator=(const RenderBackendDRM&);
};

