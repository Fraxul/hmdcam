#include "rhi/egl/RHIEGLStreamSurfaceGL.h"

RHIEGLStreamSurfaceGL::~RHIEGLStreamSurfaceGL() {

}

RHIEGLStreamSurfaceGL::RHIEGLStreamSurfaceGL(uint32_t width_, uint32_t height_, RHISurfaceFormat format_) {
  glGenTextures(1, &m_glId);

  m_glTarget = GL_TEXTURE_EXTERNAL_OES;
  m_glInternalFormat = 0; // ?
  m_width = width_;
  m_height = height_;
  m_depth = 0;
  m_layers = 1;
  m_samples = 1;
  m_levels = 1;
  m_rhiFormat = format_;
  m_isArrayTexture = false;
}

