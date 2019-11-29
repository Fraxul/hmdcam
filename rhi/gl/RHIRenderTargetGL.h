#pragma once
#include "rhi/RHIRenderTarget.h"
#include "rhi/gl/GLCommon.h"

class RHIRenderTargetGL : public RHIRenderTarget {
public:
  typedef boost::intrusive_ptr<RHIRenderTargetGL> ptr;
  virtual ~RHIRenderTargetGL();

  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t layers() const;
  virtual uint32_t samples() const;
  virtual bool isArray() const;

  virtual bool isWindowRenderTarget() const;
  virtual size_t colorTargetCount() const;
  virtual bool hasDepthStencilTarget() const;

  GLuint glFramebufferId() const { return m_glFramebufferId; }
  static RHIRenderTargetGL* newRenderTarget(const RHIRenderTargetDescriptor&);

protected:
  RHIRenderTargetGL();
  static void handleAttachment(GLenum attachPoint, const RHIRenderTargetDescriptorElement&);
  static bool formatCheck(RHIRenderTargetGL* target, const RHIRenderTargetDescriptorElement&, bool& haveFormat);

  GLuint m_glFramebufferId;
  uint32_t m_width, m_height, m_layers, m_samples;
  uint32_t m_colorTargetCount;
  bool m_isArray;
  bool m_hasDepthStencilTarget;

};
