#pragma once
#include "rhi/RHIBuffer.h"
#include "rhi/gl/GLCommon.h"

class RHIBufferGL : public RHIBuffer {
public:
  typedef boost::intrusive_ptr<RHIBufferGL> ptr;
  RHIBufferGL(GLuint, size_t, RHIBufferUsageMode);
  virtual ~RHIBufferGL();

  virtual void map(RHIBufferMapMode);
  virtual void unmap();

  GLuint glId() const { return m_buffer; }
  void bufferData(const void*, size_t);
  void bufferSubData(const void*, size_t length, size_t offset = 0);

protected:
  RHIBufferGL() = default;
  GLuint m_buffer = 0;
};
