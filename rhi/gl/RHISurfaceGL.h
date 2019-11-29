#pragma once
#include "rhi/RHISurface.h"
#include "rhi/gl/GLCommon.h"

GLenum RHISurfaceFormatToGL(RHISurfaceFormat format);

class RHISamplerGL : public RHISampler {
public:
  typedef boost::intrusive_ptr<RHISamplerGL> ptr;
  virtual ~RHISamplerGL();

  GLuint glId() const { return m_glId; }

  RHISamplerGL(const RHISamplerDescriptor&);
protected:
  GLuint m_glId;
};

class RHISurfaceGL : public RHISurface {
public:
  typedef boost::intrusive_ptr<RHISurfaceGL> ptr;
  virtual ~RHISurfaceGL();

  virtual bool isValidRenderTarget() const;
  virtual bool isValidTextureSource() const;

  virtual RHISurfaceFormat format() const;
  virtual uint32_t width() const;
  virtual uint32_t height() const;
  virtual uint32_t depth() const;
  virtual uint32_t layers() const;
  virtual uint32_t samples() const;
  virtual uint32_t mipLevels() const;
  virtual bool isArray() const;

  static RHISurfaceGL* newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&);
  static RHISurfaceGL* newTexture3D(uint32_t width, uint32_t height, uint32_t depth, const RHISurfaceDescriptor&);
  static RHISurfaceGL* newRenderbuffer2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&);

  GLuint glId() const { return m_glId; }
  GLenum glTarget() const { return m_glTarget; }
  GLenum glInternalFormat() const { return m_glInternalFormat; }

  bool isGLRenderbuffer() const { return m_glTarget == GL_RENDERBUFFER; }
  bool isGLTexture() const { return !isGLRenderbuffer(); }
  
protected:
  RHISurfaceGL();

  GLuint m_glId;
  GLenum m_glTarget;
  GLenum m_glInternalFormat;
  uint32_t m_width, m_height, m_depth;
  uint32_t m_layers, m_samples, m_levels;
  RHISurfaceFormat m_rhiFormat;
  bool m_isArrayTexture;
};
