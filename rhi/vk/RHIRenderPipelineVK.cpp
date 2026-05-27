#include "rhi/vk/RHIRenderPipelineVK.h"
#include "rhi/RHI.h"
#include "rhi/vk/RHISamplerVK.h"
#include <cstdio>
#include <cstdlib>
#include <set>

namespace {

vk::Format vkFormatForVertexElement(RHIVertexElementType t) {
  switch (t) {
    case kVertexElementTypeFloat1: return vk::Format::eR32Sfloat;
    case kVertexElementTypeFloat2: return vk::Format::eR32G32Sfloat;
    case kVertexElementTypeFloat3: return vk::Format::eR32G32B32Sfloat;
    case kVertexElementTypeFloat4: return vk::Format::eR32G32B32A32Sfloat;

    case kVertexElementTypeHalf1: return vk::Format::eR16Sfloat;
    case kVertexElementTypeHalf2: return vk::Format::eR16G16Sfloat;
    case kVertexElementTypeHalf4: return vk::Format::eR16G16B16A16Sfloat;

    case kVertexElementTypeInt1: return vk::Format::eR32Sint;
    case kVertexElementTypeInt2: return vk::Format::eR32G32Sint;
    case kVertexElementTypeInt3: return vk::Format::eR32G32B32Sint;
    case kVertexElementTypeInt4: return vk::Format::eR32G32B32A32Sint;

    case kVertexElementTypeUInt1: return vk::Format::eR32Uint;
    case kVertexElementTypeUInt2: return vk::Format::eR32G32Uint;
    case kVertexElementTypeUInt3: return vk::Format::eR32G32B32Uint;
    case kVertexElementTypeUInt4: return vk::Format::eR32G32B32A32Uint;

    case kVertexElementTypeByte1: return vk::Format::eR8Sint;
    case kVertexElementTypeByte2: return vk::Format::eR8G8Sint;
    case kVertexElementTypeByte3: return vk::Format::eR8G8B8Sint;
    case kVertexElementTypeByte4: return vk::Format::eR8G8B8A8Sint;

    case kVertexElementTypeUByte1: return vk::Format::eR8Uint;
    case kVertexElementTypeUByte2: return vk::Format::eR8G8Uint;
    case kVertexElementTypeUByte3: return vk::Format::eR8G8B8Uint;
    case kVertexElementTypeUByte4: return vk::Format::eR8G8B8A8Uint;

    case kVertexElementTypeByte1N: return vk::Format::eR8Snorm;
    case kVertexElementTypeByte2N: return vk::Format::eR8G8Snorm;
    case kVertexElementTypeByte3N: return vk::Format::eR8G8B8Snorm;
    case kVertexElementTypeByte4N: return vk::Format::eR8G8B8A8Snorm;

    case kVertexElementTypeUByte1N: return vk::Format::eR8Unorm;
    case kVertexElementTypeUByte2N: return vk::Format::eR8G8Unorm;
    case kVertexElementTypeUByte3N: return vk::Format::eR8G8B8Unorm;
    case kVertexElementTypeUByte4N: return vk::Format::eR8G8B8A8Unorm;

    case kVertexElementTypeUShort1: return vk::Format::eR16Uint;
    case kVertexElementTypeUShort2: return vk::Format::eR16G16Uint;
    case kVertexElementTypeUShort4: return vk::Format::eR16G16B16A16Uint;

    case kVertexElementTypeUShort1N: return vk::Format::eR16Unorm;
    case kVertexElementTypeUShort2N: return vk::Format::eR16G16Unorm;
    case kVertexElementTypeUShort4N: return vk::Format::eR16G16B16A16Unorm;

    case kVertexElementTypeShort1: return vk::Format::eR16Sint;
    case kVertexElementTypeShort2: return vk::Format::eR16G16Sint;
    case kVertexElementTypeShort4: return vk::Format::eR16G16B16A16Sint;

    case kVertexElementTypeShort1N: return vk::Format::eR16Snorm;
    case kVertexElementTypeShort2N: return vk::Format::eR16G16Snorm;
    case kVertexElementTypeShort4N: return vk::Format::eR16G16B16A16Snorm;

    default:
      fprintf(stderr, "RHIRenderPipelineVK: unhandled vertex element type %d\n", static_cast<int>(t));
      abort();
  }
}

vk::PrimitiveTopology vkPrimitiveTopologyFor(RHIPrimitiveTopology t) {
  switch (t) {
    case kPrimitiveTopologyPoints: return vk::PrimitiveTopology::ePointList;
    case kPrimitiveTopologyLineList: return vk::PrimitiveTopology::eLineList;
    case kPrimitiveTopologyLineStrip: return vk::PrimitiveTopology::eLineStrip;
    case kPrimitiveTopologyTriangleList: return vk::PrimitiveTopology::eTriangleList;
    case kPrimitiveTopologyTriangleStrip: return vk::PrimitiveTopology::eTriangleStrip;
    case kPrimitiveTopologyPatches: return vk::PrimitiveTopology::ePatchList;
    default:
      fprintf(stderr, "RHIRenderPipelineVK: unhandled primitive topology %d\n", static_cast<int>(t));
      abort();
  }
}

vk::ShaderStageFlagBits vkStageForShadingUnit(RHIShaderDescriptor::ShadingUnit u) {
  switch (u) {
    case RHIShaderDescriptor::kVertexShader: return vk::ShaderStageFlagBits::eVertex;
    case RHIShaderDescriptor::kFragmentShader: return vk::ShaderStageFlagBits::eFragment;
    case RHIShaderDescriptor::kGeometryShader: return vk::ShaderStageFlagBits::eGeometry;
    case RHIShaderDescriptor::kTessControlShader: return vk::ShaderStageFlagBits::eTessellationControl;
    case RHIShaderDescriptor::kTessEvaluationShader: return vk::ShaderStageFlagBits::eTessellationEvaluation;
    case RHIShaderDescriptor::kComputeShader: return vk::ShaderStageFlagBits::eCompute;
    default:
      fprintf(stderr, "RHIRenderPipelineVK: unhandled shading unit %d\n", static_cast<int>(u));
      abort();
  }
}

} // namespace

RHIRenderPipelineVK::~RHIRenderPipelineVK() = default;

RHIRenderPipelineVK::RHIRenderPipelineVK(RHIShaderVK::ptr shader, const RHIRenderPipelineDescriptor& d) :
  m_shader(shader),
  m_descriptor(d) {

  vk::Device device = rhi()->vk()->device();

  // ---- 1. Aggregate resource bindings across all stages. ----
  // For each (set, binding) we union the stage flags and verify that the
  // descriptor type agrees across stages. Same name in two stages means
  // the same binding has to use the same descriptor type — glslang/shaderc
  // assigns them sequentially, so this is the natural case.
  struct AggregatedBinding {
    uint32_t set;
    uint32_t binding;
    vk::DescriptorType descriptorType;
    vk::ShaderStageFlags stageFlags;
    uint32_t arraySize;
    std::string name;
  };
  std::map<std::pair<uint32_t, uint32_t>, AggregatedBinding> aggBindings;

  const char* dumpEnv = getenv("RHI_VK_DUMP_BINDINGS");
  if (dumpEnv && atoi(dumpEnv)) {
    fprintf(stderr, "RHIRenderPipelineVK: bindings for pipeline %p\n", static_cast<void*>(this));
    for (const auto& [unit, refl] : m_shader->stagesReflection) {
      fprintf(stderr, "  stage %s:\n", RHIShaderDescriptor::nameForShadingUnit(unit));
      for (const auto& b : refl.bindings) {
        fprintf(stderr, "    set=%u binding=%u type=%s name=\"%s\"\n",
          b.set, b.binding, vk::to_string(b.descriptorType).c_str(), b.name.c_str());
      }
      for (const auto& vi : refl.vertexInputs) {
        fprintf(stderr, "    vtx-input loc=%u name=\"%s\"\n", vi.location, vi.name.c_str());
      }
    }
  }

  for (const auto& [unit, refl] : m_shader->stagesReflection) {
    vk::ShaderStageFlagBits stage = vkStageForShadingUnit(unit);
    for (const auto& b : refl.bindings) {
      auto key = std::make_pair(b.set, b.binding);
      auto it = aggBindings.find(key);
      if (it == aggBindings.end()) {
        AggregatedBinding agg;
        agg.set = b.set;
        agg.binding = b.binding;
        agg.descriptorType = b.descriptorType;
        agg.stageFlags = stage;
        agg.arraySize = b.arraySize;
        agg.name = b.name;
        aggBindings[key] = agg;
      } else {
        it->second.stageFlags |= stage;
        if (it->second.descriptorType != b.descriptorType) {
          fprintf(stderr, "RHIRenderPipelineVK: descriptor-type mismatch at set=%u binding=%u between stages\n",
            b.set, b.binding);
        }
      }
    }
  }

  // Resolve immutable-sampler declarations (set on the shader prior to
  // pipeline build) into (binding -> VkSampler) handles. The shader carries
  // RHISampler::ptr keyed by GLSL identifier; we look those up against the
  // aggregated binding names so each immutable sampler lands on every slot
  // that names it (across stages, when our per-stage offset puts the same
  // name at multiple bindings).
  std::map<std::pair<uint32_t, uint32_t>, RHISamplerVK*> immutableByBinding;
  for (const auto& [name, samplerPtr] : m_shader->immutableSamplers()) {
    RHISamplerVK* samplerVk = static_cast<RHISamplerVK*>(samplerPtr.get());
    bool foundAny = false;
    for (const auto& [key, agg] : aggBindings) {
      if (agg.name == name) {
        immutableByBinding[key] = samplerVk;
        foundAny = true;
      }
    }
    if (!foundAny) {
      fprintf(stderr, "RHIRenderPipelineVK: immutable sampler binding \"%s\" not found in any shader stage\n", name.c_str());
    }
  }

  // Populate name → bindings map for fast load*() lookup. A single name
  // can map to multiple bindings when the same resource (typically a
  // uniform block) is referenced from multiple shader stages: our
  // per-stage binding-number offset puts each stage's reference at a
  // distinct slot. load*() fans out the descriptor write to all matching
  // slots so every stage sees the data. Typical entry has size 1.
  for (const auto& [key, agg] : aggBindings) {
    ResourceBinding rb;
    rb.set = agg.set;
    rb.binding = agg.binding;
    rb.descriptorType = agg.descriptorType;
    rb.stageFlags = agg.stageFlags;
    rb.arraySize = agg.arraySize;
    auto immIt = immutableByBinding.find(key);
    if (immIt != immutableByBinding.end())
      rb.immutableSampler = immIt->second;
    if (!agg.name.empty())
      m_resourceBindings[agg.name].push_back(rb);
  }

  // ---- 2. Build one VkDescriptorSetLayout per used `set` index. ----
  // Use the push-descriptor flag so we never have to manage a descriptor
  // pool — bindings are pushed inline per-draw. The shader-object pipeline
  // layout must include a slot for every set up to the max referenced set
  // (gaps would be illegal); we synthesize empty layouts for any holes.
  //
  // Immutable samplers must be specified via pImmutableSamplers on the
  // matching binding entry. The pointed-to VkSampler array must outlive
  // the descriptor set layout creation call (we keep a side vector per
  // binding while building, then the driver copies the handle into the
  // layout itself).
  uint32_t maxSet = 0;
  std::map<uint32_t, std::vector<vk::DescriptorSetLayoutBinding>> bindingsBySet;
  // Holds the single VkSampler each immutable binding refers to. Indexed
  // by (set, binding) so the address handed to Vulkan stays stable across
  // bindingsBySet vector reallocations.
  std::map<std::pair<uint32_t, uint32_t>, vk::Sampler> immutableSamplerHandles;
  for (const auto& [key, agg] : aggBindings) {
    vk::DescriptorSetLayoutBinding lb;
    lb.binding = agg.binding;
    lb.descriptorType = agg.descriptorType;
    lb.descriptorCount = agg.arraySize;
    lb.stageFlags = agg.stageFlags;
    auto immIt = immutableByBinding.find(key);
    if (immIt != immutableByBinding.end()) {
      immutableSamplerHandles[key] = immIt->second->vkSampler();
      lb.pImmutableSamplers = &immutableSamplerHandles[key];
    }
    bindingsBySet[agg.set].push_back(lb);
    if (agg.set > maxSet) maxSet = agg.set;
  }

  if (!aggBindings.empty()) {
    m_descriptorSetLayouts.resize(maxSet + 1);
    m_descriptorSetLayoutsRaw.resize(maxSet + 1);
    for (uint32_t s = 0; s <= maxSet; ++s) {
      vk::DescriptorSetLayoutCreateInfo lci;
      lci.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
      auto it = bindingsBySet.find(s);
      if (it != bindingsBySet.end()) {
        lci.bindingCount = static_cast<uint32_t>(it->second.size());
        lci.pBindings = it->second.data();
      }
      m_descriptorSetLayouts[s] = device.createDescriptorSetLayoutUnique(lci);
      m_descriptorSetLayoutsRaw[s] = m_descriptorSetLayouts[s].get();
    }
  }

  // ---- 3. Push-constant range (union of all stages that declared one). ----
  std::vector<vk::PushConstantRange> pushConstantRanges;
  uint32_t totalPushConstantSize = 0;
  vk::ShaderStageFlags pushConstantStages{};
  for (const auto& [unit, refl] : m_shader->stagesReflection) {
    if (refl.pushConstantSize > 0) {
      pushConstantStages |= vkStageForShadingUnit(unit);
      if (refl.pushConstantSize > totalPushConstantSize)
        totalPushConstantSize = refl.pushConstantSize;
    }
  }
  if (totalPushConstantSize > 0) {
    vk::PushConstantRange pcr;
    pcr.stageFlags = pushConstantStages;
    pcr.offset = 0;
    pcr.size = totalPushConstantSize;
    pushConstantRanges.push_back(pcr);
  }

  // ---- 4. Pipeline layout. ----
  vk::PipelineLayoutCreateInfo plci;
  plci.setLayoutCount = static_cast<uint32_t>(m_descriptorSetLayoutsRaw.size());
  plci.pSetLayouts = m_descriptorSetLayoutsRaw.data();
  plci.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
  plci.pPushConstantRanges = pushConstantRanges.data();
  m_pipelineLayout = device.createPipelineLayoutUnique(plci);

  // ---- 5. VkShaderEXTs, linked. ----
  // Linked stages need to be created in a single createShadersEXT call so
  // the implementation can do whole-program optimization across stages.
  std::vector<vk::ShaderCreateInfoEXT> shaderCis;
  shaderCis.reserve(m_shader->stagesSpirv.size());
  m_shaderStages.reserve(m_shader->stagesSpirv.size());

  // Order matters: we want next-stage bits set correctly. Build vertex
  // first, then geometry/tess if present, then fragment.
  static constexpr RHIShaderDescriptor::ShadingUnit kStageOrder[] = {
    RHIShaderDescriptor::kVertexShader,
    RHIShaderDescriptor::kTessControlShader,
    RHIShaderDescriptor::kTessEvaluationShader,
    RHIShaderDescriptor::kGeometryShader,
    RHIShaderDescriptor::kFragmentShader,
  };
  std::vector<RHIShaderDescriptor::ShadingUnit> presentStages;
  for (auto unit : kStageOrder) {
    if (m_shader->stagesSpirv.count(unit))
      presentStages.push_back(unit);
  }

  for (size_t i = 0; i < presentStages.size(); ++i) {
    RHIShaderDescriptor::ShadingUnit unit = presentStages[i];
    const auto& spirv = m_shader->stagesSpirv[unit];

    vk::ShaderCreateInfoEXT ci;
    ci.flags = vk::ShaderCreateFlagBitsEXT::eLinkStage;
    ci.stage = vkStageForShadingUnit(unit);
    if (i + 1 < presentStages.size())
      ci.nextStage = static_cast<vk::ShaderStageFlags>(vkStageForShadingUnit(presentStages[i + 1]));
    ci.codeType = vk::ShaderCodeTypeEXT::eSpirv;
    ci.codeSize = spirv.size() * sizeof(uint32_t);
    ci.pCode = spirv.data();
    ci.pName = "main";
    ci.setLayoutCount = static_cast<uint32_t>(m_descriptorSetLayoutsRaw.size());
    ci.pSetLayouts = m_descriptorSetLayoutsRaw.data();
    ci.pushConstantRangeCount = static_cast<uint32_t>(pushConstantRanges.size());
    ci.pPushConstantRanges = pushConstantRanges.data();
    shaderCis.push_back(ci);
    m_shaderStages.push_back(vkStageForShadingUnit(unit));
  }

  if (!shaderCis.empty()) {
    try {
      m_shaderObjects = device.createShadersEXTUnique(shaderCis);
    } catch (const vk::SystemError& ex) {
      fprintf(stderr, "RHIRenderPipelineVK: createShadersEXT failed: %s\n", ex.what());
      abort();
    }
    m_shaderObjectsRaw.reserve(m_shaderObjects.size());
    for (size_t i = 0; i < m_shaderObjects.size(); ++i) {
      vk::ShaderEXT raw = m_shaderObjects[i].get();
      if (!raw) {
        fprintf(stderr, "RHIRenderPipelineVK: stage %s ended up as VK_NULL_HANDLE after createShadersEXT (silent partial failure)\n",
          vk::to_string(m_shaderStages[i]).c_str());
        abort();
      }
      m_shaderObjectsRaw.push_back(raw);
    }
  }

  // ---- 6. Vertex input: match RHIVertexLayout against the vertex stage's
  // Input variables (by name) to produce the vkCmdSetVertexInputEXT arrays.
  // For stream buffers that don't appear in the shader (e.g. instance-rate
  // attributes used only by some pipelines), we silently drop them.
  auto vsReflIt = m_shader->stagesReflection.find(RHIShaderDescriptor::kVertexShader);
  if (vsReflIt != m_shader->stagesReflection.end()) {
    const SpirvReflection& vsRefl = vsReflIt->second;
    std::map<std::string, uint32_t> nameToLocation;
    for (const auto& vi : vsRefl.vertexInputs)
      nameToLocation[vi.name] = vi.location;

    // Track which stream buffers (input bindings) are actually referenced
    // so we emit each binding exactly once with its stride.
    std::set<uint8_t> usedStreams;
    for (const auto& el : m_shader->vertexLayout().elements) {
      auto locIt = nameToLocation.find(el.elementName.c_str());
      if (locIt == nameToLocation.end())
        continue; // attribute declared on host side but not referenced by the shader

      vk::VertexInputAttributeDescription2EXT attr;
      attr.location = locIt->second;
      attr.binding = el.streamBufferIndex;
      attr.format = vkFormatForVertexElement(el.elementType);
      attr.offset = el.offset;
      m_vertexAttributes.push_back(attr);

      if (usedStreams.insert(el.streamBufferIndex).second) {
        vk::VertexInputBindingDescription2EXT bind;
        bind.binding = el.streamBufferIndex;
        bind.stride = el.stride;
        bind.inputRate = (el.elementFrequency == kVertexElementFrequencyInstance)
          ? vk::VertexInputRate::eInstance
          : vk::VertexInputRate::eVertex;
        bind.divisor = 1;
        m_vertexBindings.push_back(bind);
      }
    }
  }

  m_primitiveTopology = vkPrimitiveTopologyFor(m_descriptor.primitiveTopology);

  if (dumpEnv && atoi(dumpEnv)) {
    for (const auto& b : m_vertexBindings) {
      fprintf(stderr, "    vtx-binding %u stride=%u rate=%s\n", b.binding, b.stride,
        b.inputRate == vk::VertexInputRate::eVertex ? "vertex" : "instance");
    }
    for (const auto& a : m_vertexAttributes) {
      fprintf(stderr, "    vtx-attr loc=%u binding=%u format=%s offset=%u\n",
        a.location, a.binding, vk::to_string(a.format).c_str(), a.offset);
    }
  }
}
