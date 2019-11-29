#pragma once
#include "rhi/RHIBuffer.h"
#include "rhi/gl/GLCommon.h"

class RHIBufferGL : public RHIBuffer {
public:
  typedef boost::intrusive_ptr<RHIBufferGL> ptr;
  RHIBufferGL();
  RHIBufferGL(GLuint, size_t, RHIBufferUsageMode);
  virtual ~RHIBufferGL();

  virtual void map(RHIBufferMapMode);
  virtual void unmap();

  GLuint glId() const { return m_buffer; }
  void bufferData(const void*, size_t);
protected:
  GLuint m_buffer;
};

