#pragma once
// Minimal SPIR-V reflection used by RHIShaderVK to derive descriptor
// bindings, vertex input locations, and push-constant size from the
// shaderc-emitted SPIR-V. Replaces the role SPIRV-Cross would play in a
// fuller toolchain; we only need the subset of opcodes the engine's
// shaders actually use.
//
// The reflector is a single-pass walk of the SPIR-V word stream that
// records: type definitions (pointer, image, sampled-image, struct),
// names (OpName / OpMemberName), decorations (DescriptorSet, Binding,
// Block, BufferBlock, Location), and variable declarations (OpVariable).
// Resource kinds are then classified by combining the variable's storage
// class with its pointee type and any decorations.

#include "rhi/vk/RHIVulkan.h"
#include <cstdint>
#include <string>
#include <vector>

struct SpirvDescriptorBinding {
  std::string name;
  uint32_t set = 0;
  uint32_t binding = 0;
  vk::DescriptorType descriptorType{};
  uint32_t arraySize = 1; // 1 == not declared as an array
};

struct SpirvVertexInput {
  std::string name;
  uint32_t location = 0;
};

struct SpirvReflection {
  std::vector<SpirvDescriptorBinding> bindings;
  std::vector<SpirvVertexInput> vertexInputs;
  uint32_t pushConstantSize = 0; // 0 if no PushConstant variable
};

// Parse the SPIR-V module and return the reflected metadata. Aborts on a
// malformed header — anything past the header is best-effort: unknown
// opcodes are skipped via the length field.
SpirvReflection reflectSpirv(const std::vector<uint32_t>& spirv);
