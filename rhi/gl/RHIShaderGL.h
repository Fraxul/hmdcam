#pragma once
#include "rhi/gl/GLCommon.h"
#include "rhi/RHIShader.h"

class RHIGL;

class RHIShaderGL : public RHIShader {
public:
  typedef boost::intrusive_ptr<RHIShaderGL> ptr;
  virtual ~RHIShaderGL();

  int32_t uniformBlockLocation(const FxAtomicString&);
  int32_t varyingAttributeLocation(const FxAtomicString&);
  // actually returns the texture unit number, since unit bindings are done at compile time.
  int32_t samplerAttributeLocation(const FxAtomicString&);
  int32_t bufferBlockLocation(const FxAtomicString&);

  const RHIVertexLayout& vertexLayout() const { return m_vertexLayout; }

  GLuint program() const { return m_program; }

protected:
  friend class RHIGL;
  RHIShaderGL(const RHIShaderDescriptor&);
  GLuint compileShader(RHIShaderDescriptor::ShadingUnit type, const char* source);
  void internalHandleLinkFailure();

  struct Attribute {
    FxAtomicString name;
    GLenum type;
    GLint location;
    GLint textureUnit;
  };

  GLuint m_program;
  RHIShaderDescriptor m_descriptor;
  RHIVertexLayout m_vertexLayout;
  std::vector<Attribute> m_varyingAttributes;
  std::vector<Attribute> m_samplerAttributes;
  std::vector<Attribute> m_uniformBlocks;
  std::vector<Attribute> m_bufferBlocks;
};

