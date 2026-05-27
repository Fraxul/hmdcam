#pragma once
// RHIShaderVK: Vulkan implementation of RHIShader.
//
// Holds per-stage SPIR-V + the reflection (descriptor bindings, vertex
// input locations, push-constant size) produced once at compile time.
// RHIRenderPipelineVK reads from these tables to build VkShaderEXTs and
// VkDescriptorSetLayouts; RHIVK consults them during load*/draw* to
// resolve FxAtomicString names back to (set, binding) pairs.

#include "rhi/RHIShader.h"
#include "rhi/vk/SpirvReflect.h"
#include <map>
#include <vector>

class RHIShaderVK : public RHIShader {
public:
  typedef boost::intrusive_ptr<RHIShaderVK> ptr;

  RHIShaderVK();
  virtual ~RHIShaderVK();

  // Per-stage compiled SPIR-V. Indexed by the descriptor's ShadingUnit enum.
  std::map<RHIShaderDescriptor::ShadingUnit, std::vector<uint32_t>> stagesSpirv;

  // Per-stage reflection produced by reflectSpirv() at compile time.
  std::map<RHIShaderDescriptor::ShadingUnit, SpirvReflection> stagesReflection;

  // Vertex layout: copied from the descriptor at compile time so the pipeline
  // can reconstruct vertex input bindings without retaining the descriptor.
  // Matches the pattern used in RHIShaderGL.
  const RHIVertexLayout& vertexLayout() const { return m_vertexLayout; }

  // Compile every stage in the descriptor through shaderc, return a populated
  // RHIShaderVK. Aborts on compilation failure with a diagnostic dump.
  static RHIShader::ptr compileFromDescriptor(const RHIShaderDescriptor&);

protected:
  RHIVertexLayout m_vertexLayout;
};
