#include "rhi/gl/RHIEGLSurfaceRenderTargetGL.h"

RHIEGLSurfaceRenderTargetGL::RHIEGLSurfaceRenderTargetGL(EGLDisplay display_, EGLSurface surface_) : m_display(display_), m_surface(surface_) {


}

RHIEGLSurfaceRenderTargetGL::~RHIEGLSurfaceRenderTargetGL() {

}

void RHIEGLSurfaceRenderTargetGL::platformSwapBuffers() {
  eglSwapBuffers(display(), surface());
}

