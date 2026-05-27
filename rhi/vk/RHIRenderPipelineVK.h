#pragma once
// RHIRenderPipelineVK: render-pipeline state for the VK backend.
//
// With VK_EXT_shader_object there is no monolithic VkPipeline for
// graphics — instead each shader stage is its own VkShaderEXT, bound
// independently, with all other pipeline state (vertex input, blend,
// depth, raster, primitive topology, render target format) set
// dynamically per draw via vkCmdSet*EXT.
//
// What we still need ahead of time:
//   - VkDescriptorSetLayout per descriptor set referenced by any stage.
//   - VkPipelineLayout consolidating the set layouts and push-constant
//     range (so VkShaderEXTs are created against a known interface).
//   - The linked VkShaderEXTs themselves.
//   - Pre-computed vertex input arrays for vkCmdSetVertexInputEXT,
//     resolved from RHIVertexLayout + vertex-shader Input reflection.
//
// Pipelines are built immediately upon construction, to avoid latency spikes
// during bind / draw operations.

#include "rhi/RHIRenderPipeline.h"
#include "rhi/vk/RHIShaderVK.h"
#include "rhi/vk/RHIVulkan.h"
#include <map>
#include <vector>

class RHIRenderPipelineVK : public RHIRenderPipeline {
public:
  typedef boost::intrusive_ptr<RHIRenderPipelineVK> ptr;

  RHIRenderPipelineVK(RHIShaderVK::ptr shader, const RHIRenderPipelineDescriptor& d);

  virtual ~RHIRenderPipelineVK();

  virtual RHIShader* shader() const override { return m_shader.get(); }
  RHIShaderVK* shaderVK() const { return m_shader.get(); }

  const RHIRenderPipelineDescriptor& descriptor() const { return m_descriptor; }
  const RHIVertexLayout& vertexLayout() const { return m_shader->vertexLayout(); }

  // Resource-binding metadata, populated by build(). Indexed by descriptor
  // FxAtomicString name (matched against the GLSL identifier). For
  // resolved name → binding lookups at load*Texture/UniformBlock/etc. time.
  struct ResourceBinding {
    uint32_t set = 0;
    uint32_t binding = 0;
    vk::DescriptorType descriptorType{};
    vk::ShaderStageFlags stageFlags{};
    uint32_t arraySize = 1;
    // Non-owning. Non-null iff this binding was declared with
    // RHIShader::setImmutableSampler. The sampler handle is baked into the
    // descriptor set layout's pImmutableSamplers; loadTexture treats the
    // sampler half of the descriptor write as ignored for these bindings
    // (the spec says immutable samplers shadow the write's sampler).
    class RHISamplerVK* immutableSampler = nullptr;
  };
  // Keyed by name (case-sensitive; matches the FxAtomicString the engine
  // uses in load*). The value is a vector because the same uniform block
  // can legitimately be referenced from multiple stages — and because
  // our per-stage binding-number offset (RHIShaderVK::stageBindingOffset)
  // puts each stage's reference at a different binding slot, a single
  // load*() needs to fan out to all of them. Typical vector size is 1.
  const std::map<std::string, std::vector<ResourceBinding>>& resourceBindings() const { return m_resourceBindings; }

  // Vertex-input arrays in the exact form vkCmdSetVertexInputEXT expects.
  const std::vector<vk::VertexInputBindingDescription2EXT>& vertexBindings() const { return m_vertexBindings; }
  const std::vector<vk::VertexInputAttributeDescription2EXT>& vertexAttributes() const { return m_vertexAttributes; }

  vk::PipelineLayout pipelineLayout() const { return m_pipelineLayout.get(); }
  const std::vector<vk::DescriptorSetLayout>& descriptorSetLayouts() const { return m_descriptorSetLayoutsRaw; }
  vk::PrimitiveTopology primitiveTopology() const { return m_primitiveTopology; }

  // Shader objects in bind order; pair with m_shaderStages for vkCmdBindShadersEXT.
  const std::vector<vk::ShaderEXT>& shaderObjects() const { return m_shaderObjectsRaw; }
  const std::vector<vk::ShaderStageFlagBits>& shaderStages() const { return m_shaderStages; }

protected:
  RHIShaderVK::ptr m_shader;
  RHIRenderPipelineDescriptor m_descriptor;

  std::vector<vk::UniqueDescriptorSetLayout> m_descriptorSetLayouts;
  std::vector<vk::DescriptorSetLayout> m_descriptorSetLayoutsRaw;
  vk::UniquePipelineLayout m_pipelineLayout;
  std::vector<vk::UniqueShaderEXT> m_shaderObjects;
  std::vector<vk::ShaderEXT> m_shaderObjectsRaw;
  std::vector<vk::ShaderStageFlagBits> m_shaderStages;

  std::map<std::string, std::vector<ResourceBinding>> m_resourceBindings;
  std::vector<vk::VertexInputBindingDescription2EXT> m_vertexBindings;
  std::vector<vk::VertexInputAttributeDescription2EXT> m_vertexAttributes;
  vk::PrimitiveTopology m_primitiveTopology = vk::PrimitiveTopology::eTriangleList;
};
