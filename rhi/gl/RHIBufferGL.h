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

  virtual CUgraphicsResource& cuGraphicsResource() const;

  GLuint glId() const { return m_buffer; }
  void bufferData(const void*, size_t);
protected:
  GLuint m_buffer;
  mutable CUgraphicsResource m_cuGraphicsResource;
};

