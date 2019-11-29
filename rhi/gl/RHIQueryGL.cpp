#include "rhi/gl/RHIQueryGL.h"

RHITimerQueryGL::RHITimerQueryGL() : m_glId(0) {
  glGenQueries(1, &m_glId);
}

RHITimerQueryGL::~RHITimerQueryGL() {
  glDeleteQueries(1, &m_glId);
}

RHIOcclusionQueryGL::RHIOcclusionQueryGL(RHIOcclusionQueryMode mode) : m_glId(0), m_mode(mode) {
  glGenQueries(1, &m_glId);
}

RHIOcclusionQueryGL::~RHIOcclusionQueryGL() {
  glDeleteQueries(1, &m_glId);
}

