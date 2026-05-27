#include "rhi/vk/RHIShaderVK.h"
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <shaderc/shaderc.hpp>

namespace {

// Long-lived shaderc compiler. Construction allocates glslang internal state;
// keep one instance for the process lifetime rather than per-shader.
shaderc::Compiler& sharedCompiler() {
  static shaderc::Compiler compiler;
  return compiler;
}

shaderc_shader_kind shadercKindForUnit(RHIShaderDescriptor::ShadingUnit unit) {
  switch (unit) {
    case RHIShaderDescriptor::kVertexShader: return shaderc_glsl_vertex_shader;
    case RHIShaderDescriptor::kFragmentShader: return shaderc_glsl_fragment_shader;
    case RHIShaderDescriptor::kGeometryShader: return shaderc_glsl_geometry_shader;
    case RHIShaderDescriptor::kTessControlShader: return shaderc_glsl_tess_control_shader;
    case RHIShaderDescriptor::kTessEvaluationShader: return shaderc_glsl_tess_evaluation_shader;
    case RHIShaderDescriptor::kComputeShader: return shaderc_glsl_compute_shader;
    default:
      fprintf(stderr, "RHIShaderVK: unhandled ShadingUnit %d\n", static_cast<int>(unit));
      abort();
  }
}

// Vulkan target needs core profile, not ES — gl_ViewportIndex from arbitrary
// stages, image format layouts, geometry shaders, etc. are all core in 4.50
// but extension-gated in ES <3.20 (and some still not in 3.20).
// Also inject `#define RHI_TARGET_VULKAN 1` immediately after the version
// line so commonDeclarations.glsl's VK-target branch activates.
std::string vkPreprocess(const std::string& source) {
  size_t versionLoc = source.find("#version");
  if (versionLoc == std::string::npos) {
    fprintf(stderr, "RHIShaderVK: source has no #version directive\n");
    abort();
  }
  size_t versionLineEnd = source.find('\n', versionLoc);
  std::string versionLine = source.substr(versionLoc, (versionLineEnd + 1) - versionLoc);
  std::string body = source.substr(versionLineEnd + 1);

  if (versionLine.find("310 es") != std::string::npos) {
    size_t pos = versionLine.find("310 es");
    versionLine.replace(pos, 6, "450 core");
  }

  return source.substr(0, versionLoc) + versionLine + "#define RHI_TARGET_VULKAN 1\n" + body;
}

// Each stage gets its own contiguous chunk of binding numbers in set 0.
// 64 bindings per stage is well over what any shader in this engine
// declares; the offset goes through a post-compile SPIR-V rewrite
// rather than shaderc's SetBindingBaseForStage because the latter
// doesn't reliably catch every GLSL resource kind (combined image
// samplers in particular landed in a different slot than expected).
constexpr uint32_t kBindingsPerStage = 64;

uint32_t stageBindingOffset(RHIShaderDescriptor::ShadingUnit unit) {
  switch (unit) {
    case RHIShaderDescriptor::kVertexShader: return 0;
    case RHIShaderDescriptor::kFragmentShader: return 1 * kBindingsPerStage;
    case RHIShaderDescriptor::kGeometryShader: return 2 * kBindingsPerStage;
    case RHIShaderDescriptor::kTessControlShader: return 3 * kBindingsPerStage;
    case RHIShaderDescriptor::kTessEvaluationShader: return 4 * kBindingsPerStage;
    case RHIShaderDescriptor::kComputeShader: return 0; // compute is always alone
    default: return 0;
  }
}

shaderc::CompileOptions makeOptions() {
  shaderc::CompileOptions options;
  options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
  options.SetTargetSpirv(shaderc_spirv_version_1_6);
  options.SetSourceLanguage(shaderc_source_language_glsl);
  options.SetOptimizationLevel(shaderc_optimization_level_performance);
  // Preserve OpName / OpMemberName decorations through optimization;
  // without them the SPIR-V reflector can't recover the GLSL identifier
  // for a uniform / sampler, and load*() name-lookups silently fall
  // through to "binding not found." Performance impact is negligible —
  // debug info is metadata, not executed code.
  options.SetGenerateDebugInfo();
  // GL-style uniform blocks/samplers carry no explicit (set, binding); let
  // shaderc auto-assign sequentially from 0 per (kind). We then shift
  // every binding by a per-stage offset in SPIR-V so multi-stage
  // pipelines don't have stages stepping on each other's slots.
  options.SetAutoBindUniforms(true);
  options.SetAutoMapLocations(true);
  return options;
}

// Add `offset` to every `OpDecorate <id> Binding <literal>` in the
// SPIR-V module. Edits in place; cheap single pass over the word stream.
void offsetBindings(std::vector<uint32_t>& spirv, uint32_t offset) {
  if (offset == 0) return;
  constexpr uint32_t kOpDecorate = 71;
  constexpr uint32_t kDecorationBinding = 33;
  size_t cursor = 5; // skip 5-word header
  while (cursor < spirv.size()) {
    uint32_t word0 = spirv[cursor];
    uint32_t opcode = word0 & 0xFFFFu;
    uint32_t length = word0 >> 16;
    if (length == 0 || cursor + length > spirv.size()) break;
    if (opcode == kOpDecorate && length == 4 && spirv[cursor + 2] == kDecorationBinding) {
      spirv[cursor + 3] += offset;
    }
    cursor += length;
  }
}

} // namespace

RHIShaderVK::RHIShaderVK() {}
RHIShaderVK::~RHIShaderVK() {}

/*static*/ RHIShader::ptr RHIShaderVK::compileFromDescriptor(const RHIShaderDescriptor& descriptor) {
  std::map<RHIShaderDescriptor::ShadingUnit, std::string> sources = descriptor.preprocessSource();
  if (sources.empty()) {
    fprintf(stderr, "RHIShaderVK: descriptor produced no shading units\n");
    abort();
  }

  RHIShaderVK::ptr shader(new RHIShaderVK());
  shader->m_vertexLayout = descriptor.vertexLayout();
  shaderc::CompileOptions options = makeOptions();

  for (const auto& [unit, glslSource] : sources) {
    std::string vkSource = vkPreprocess(glslSource);
    const char* unitName = RHIShaderDescriptor::nameForShadingUnit(unit);

    shaderc::SpvCompilationResult result = sharedCompiler().CompileGlslToSpv(
      vkSource, shadercKindForUnit(unit), unitName, options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success) {
      fprintf(stderr, "RHIShaderVK: compile failed for %s:\n%s\n",
        unitName, result.GetErrorMessage().c_str());
      // Helpful follow-up: dump the actual source we sent to the compiler
      // so line numbers in the error message line up with what's diagnosed.
      RHIShaderDescriptor::dumpFormattedShaderSource(vkSource.c_str());
      abort();
    }

    shader->stagesSpirv[unit].assign(result.cbegin(), result.cend());
    // Shift this stage's binding numbers into its allocated slice of set 0
    // so cross-stage (set, binding) collisions can't happen. Must happen
    // before reflection.
    offsetBindings(shader->stagesSpirv[unit], stageBindingOffset(unit));
    // Reflect the SPIR-V right after compile so the result is ready for
    // RHIRenderPipelineVK to consume. Cheap to do here — single pass over
    // a kilobyte-sized blob — and avoids re-walking it on every bind.
    shader->stagesReflection[unit] = reflectSpirv(shader->stagesSpirv[unit]);
  }

  return shader;
}
