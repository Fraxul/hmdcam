#include "RenderBackendSurfaceless.h"
#include <epoxy/egl.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <unistd.h>

#define EGL_CHECK_BOOL(x) if (!(x)) { fprintf(stderr, "%s:%d: %s failed (%d)\n", __FILE__, __LINE__, #x, eglGetError()); abort(); }
#define CheckExtension(extStr) if (!epoxy_has_egl_extension(m_eglDisplay, extStr)) { fprintf(stderr, "Missing required EGL extension %s.\n\nExtension string: %s\n\n", extStr, eglQueryString(m_eglDisplay, EGL_EXTENSIONS)); abort(); }

RenderBackend* createSurfacelessBackend() { return new RenderBackendSurfaceless(); }

RenderBackendSurfaceless::RenderBackendSurfaceless() { }

void RenderBackendSurfaceless::init() {

  // Set up the EGL display
  {
    EGLint attrs[] = {
      EGL_NONE
    };
    EGL_CHECK_BOOL(m_eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, m_eglDevice, attrs));

    EGLint major, minor;
    EGL_CHECK_BOOL(eglInitialize(m_eglDisplay, &major, &minor));
  }

  // printf("Display Extensions: %s\n\n", eglQueryString(m_eglDisplay, EGL_EXTENSIONS));
  // printf("Device Extensions: %s\n\n", eglQueryDeviceStringEXT(m_eglDevice, EGL_EXTENSIONS));

  EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
  eglBindAPI(EGL_OPENGL_ES_API);
  EGL_CHECK_BOOL(m_eglContext = eglCreateContext(m_eglDisplay, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctx_attr));

  // Activate the display and context to make sure that libepoxy reads the right extension strings.
  // (It uses the display returned by eglGetCurrentDisplay() to detect extensions on first use)
  // If we don't do this, then loading EGL_KHR_stream and EGL_EXT_output_base will fail.
  EGL_CHECK_BOOL(eglMakeCurrent(m_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, m_eglContext));


  EGLint cfg_attr[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
    EGL_RED_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_BLUE_SIZE, 8,
    EGL_NONE
  };    

  int numConfigs;
  EGL_CHECK_BOOL(eglChooseConfig(m_eglDisplay, cfg_attr, &m_eglConfig, 1, &numConfigs));

  // 1280x720 is the expected screen size of the Monado simulated HMD
  m_surfaceWidth = 1280;
  m_surfaceHeight = 720;

  EGLint pbuffer_attr[] = {
    EGL_WIDTH, (EGLint) m_surfaceWidth,
    EGL_HEIGHT, (EGLint) m_surfaceHeight,
    EGL_NONE,
  };

  EGL_CHECK_BOOL(m_eglSurface = eglCreatePbufferSurface(m_eglDisplay, m_eglConfig, pbuffer_attr));
  EGL_CHECK_BOOL(eglMakeCurrent(m_eglDisplay, m_eglSurface, m_eglSurface, m_eglContext));

  m_windowRenderTarget = new RHIEGLSurfaceRenderTargetGL(m_eglDisplay, m_eglSurface);
  m_windowRenderTarget->platformSetUpdatedWindowDimensions(surfaceWidth(), surfaceHeight());
}

RenderBackendSurfaceless::~RenderBackendSurfaceless() {
  eglMakeCurrent(m_eglDisplay, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  eglDestroyContext(m_eglDisplay, m_eglContext);
  eglDestroySurface(m_eglDisplay, m_eglSurface);

  eglTerminate(m_eglDisplay);

  m_eglDisplay = NULL;
  m_eglSurface = NULL;
  m_eglContext = NULL;
}



