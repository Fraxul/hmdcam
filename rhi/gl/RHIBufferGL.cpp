#include "rhi/gl/RHIBufferGL.h"
#include <cuda.h>
#include <cudaGL.h>
#include "rhi/cuda/CudaUtil.h"

RHIBufferGL::RHIBufferGL() : m_buffer(0), m_cuGraphicsResource(NULL) {
  m_size = 0;
  glGenBuffers(1, &m_buffer);
}

RHIBufferGL::RHIBufferGL(GLuint bufferId, size_t size, RHIBufferUsageMode usageMode_) : m_buffer(bufferId), m_cuGraphicsResource(NULL) {
  m_size = size;
  m_usageMode = usageMode_;
}

RHIBufferGL::~RHIBufferGL() {
  if (m_cuGraphicsResource)
    cuGraphicsUnregisterResource(m_cuGraphicsResource);

  glDeleteBuffers(1, &m_buffer);
}

void RHIBufferGL::map(RHIBufferMapMode mapMode) {
  assert(usageMode() != kBufferUsageGPUPrivate && "RHIBufferGL::map(): can't map GPU-private buffers");

  if (m_data)
    return; // already mapped?

  GLenum glMapMode = GL_MAP_READ_BIT;
  switch (mapMode) {
    case kBufferMapReadOnly:
      glMapMode = GL_MAP_READ_BIT; break;
    case kBufferMapReadWrite:
      glMapMode = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT; break;
    case kBufferMapWriteOnly:
      glMapMode = GL_MAP_WRITE_BIT; break;
  };

  GL(glBindBuffer(GL_ARRAY_BUFFER, m_buffer));
  m_data = GL(glMapBufferRange(GL_ARRAY_BUFFER, 0, m_size, glMapMode));
  assert(m_data != NULL);
}

void RHIBufferGL::unmap() {
  if (!m_data)
    return;

  m_data = NULL;
  GL(glBindBuffer(GL_ARRAY_BUFFER, m_buffer));
  GL(glUnmapBuffer(GL_ARRAY_BUFFER));
}

void RHIBufferGL::bufferData(const void* data, size_t size) {
  if (m_data)
    unmap();

  GL(glBindBuffer(GL_ARRAY_BUFFER, m_buffer));
  GL(glBufferData(GL_ARRAY_BUFFER, size, data, GL_STREAM_DRAW));
  m_size = size;
}

void RHIBufferGL::bufferSubData(const void* data, size_t size, size_t offset) {
  if (m_data)
    unmap();

  GL(glBindBuffer(GL_ARRAY_BUFFER, m_buffer));
  GL(glBufferSubData(GL_ARRAY_BUFFER, offset, size, data));
  m_size = size;
}

CUgraphicsResource& RHIBufferGL::cuGraphicsResource() const {
  if (!m_cuGraphicsResource) {
    CUDA_CHECK(cuGraphicsGLRegisterBuffer(&m_cuGraphicsResource, glId(), CU_GRAPHICS_REGISTER_FLAGS_NONE));
  }

  return m_cuGraphicsResource;
}

