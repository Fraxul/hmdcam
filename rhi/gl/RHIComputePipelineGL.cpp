#include "rhi/gl/RHIComputePipelineGL.h"

RHIComputePipelineGL::RHIComputePipelineGL(RHIShaderGL::ptr shader) : m_shader(shader) {
  assert(shader);
}

RHIComputePipelineGL::~RHIComputePipelineGL() {

}

RHIShader* RHIComputePipelineGL::shader() const {
  return m_shader.get();
}


