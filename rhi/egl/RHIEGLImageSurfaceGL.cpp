#include "rhi/egl/RHIEGLImageSurfaceGL.h"

RHIEGLImageSurfaceGL::~RHIEGLImageSurfaceGL() {

}

/*static*/ RHIEGLImageSurfaceGL::ptr RHIEGLImageSurfaceGL::newTextureExternalOES(EGLImage img, uint32_t width, uint32_t height, RHISurfaceFormat reportedRHIFormat) {
  RHIEGLImageSurfaceGL* res = new RHIEGLImageSurfaceGL(img, GL_TEXTURE_EXTERNAL_OES);
  res->m_rhiFormat = reportedRHIFormat;
  res->m_width = width;
  res->m_height = height;
  return res;
}

RHIEGLImageSurfaceGL::RHIEGLImageSurfaceGL(EGLImage img, GLenum target) : m_eglImage(img) {

  m_glTarget = target;

  glGenTextures(1, &m_glId);
  glBindTexture(m_glTarget, m_glId);
  GL(glEGLImageTargetTexStorageEXT(m_glTarget, m_eglImage, /*attrList=*/ NULL));
}

void RHIEGLImageSurfaceGL::internalQueryTextureInfo() {
  glBindTexture(m_glTarget, m_glId);
  GL(glGetTexLevelParameteriv(m_glTarget, 0, GL_TEXTURE_WIDTH, (GLint*) &m_width));
  GL(glGetTexLevelParameteriv(m_glTarget, 0, GL_TEXTURE_HEIGHT, (GLint*) &m_height));
  GL(glGetTexLevelParameteriv(m_glTarget, 0, GL_TEXTURE_DEPTH, (GLint*) &m_depth));
  GL(glGetTexLevelParameteriv(m_glTarget, 0, GL_TEXTURE_INTERNAL_FORMAT, (GLint*) &m_glInternalFormat));
  GL(glGetTexLevelParameteriv(m_glTarget, 0, GL_TEXTURE_SAMPLES, (GLint*) &m_samples));
}

