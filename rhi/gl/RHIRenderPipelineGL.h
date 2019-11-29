#pragma once
#include "rhi/RHIRenderPipeline.h"
#include "rhi/gl/RHIShaderGL.h"
#include "rhi/gl/GLCommon.h"

class RHIGL;
class RHIRenderPipelineGL : public RHIRenderPipeline {
public:
  typedef boost::intrusive_ptr<RHIRenderPipelineGL> ptr;

  RHIRenderPipelineGL(RHIShaderGL::ptr, const RHIRenderPipelineDescriptor&);
  virtual ~RHIRenderPipelineGL();

  virtual RHIShader* shader() const;
  RHIShaderGL* shaderGL() const { return m_shader.get(); }

  const RHIRenderPipelineDescriptor& descriptor() const { return m_descriptor; }

  struct StreamBufferDescriptor {
    StreamBufferDescriptor() : index(-1), stride(-1), elementFrequency(kVertexElementFrequencyVertex) {}

    int index;
    int stride;
    RHIVertexElementFrequency elementFrequency;
  };

  const std::vector<StreamBufferDescriptor>& streamBufferDescriptors() const { return m_streamBufferDescriptors; }

  GLuint vao();

protected:
  RHIShaderGL::ptr m_shader;
  RHIRenderPipelineDescriptor m_descriptor;
  std::vector<StreamBufferDescriptor> m_streamBufferDescriptors;
  GLuint m_vao;
};

