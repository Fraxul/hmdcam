#pragma once
#include "rhi/RHIComputePipeline.h"
#include "rhi/gl/RHIShaderGL.h"
#include "rhi/gl/GLCommon.h"

class RHIGL;
class RHIComputePipelineGL : public RHIComputePipeline {
public:
  typedef boost::intrusive_ptr<RHIComputePipelineGL> ptr;

  RHIComputePipelineGL(RHIShaderGL::ptr);
  virtual ~RHIComputePipelineGL();

  virtual RHIShader* shader() const;
  RHIShaderGL* shaderGL() const { return m_shader.get(); }

protected:
  friend class RHIGL;
  RHIShaderGL::ptr m_shader;
};


