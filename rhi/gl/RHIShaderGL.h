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
  int32_t bufferBlockLocation(const FxAtomicString&);

  bool samplerAttributeBinding(const FxAtomicString&, uint32_t& outLocation, uint32_t& outTextureUnitNumber);
  bool imageAttributeBinding(const FxAtomicString&, uint32_t& outLocation, uint32_t& outImageUnitNumber);

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
  std::vector<Attribute> m_imageAttributes;
  std::vector<Attribute> m_uniformBlocks;
  std::vector<Attribute> m_bufferBlocks;
};

