#include "rhi/vk/SpirvReflect.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

// SPIR-V opcode + decoration + storage class IDs we care about. Values
// from the SPIR-V Unified Specification §3. Keeping these inline avoids
// pulling in <spirv/unified1/spirv.h> just for a half-dozen constants.

namespace {

constexpr uint32_t kSpirvMagic = 0x07230203u;

// Opcodes
constexpr uint32_t kOpName = 5;
constexpr uint32_t kOpMemberName = 6;
constexpr uint32_t kOpDecorate = 71;
constexpr uint32_t kOpMemberDecorate = 72;
constexpr uint32_t kOpTypeInt = 21;
constexpr uint32_t kOpTypeFloat = 22;
constexpr uint32_t kOpTypeVector = 23;
constexpr uint32_t kOpTypeMatrix = 24;
constexpr uint32_t kOpTypeImage = 25;
constexpr uint32_t kOpTypeSampler = 26;
constexpr uint32_t kOpTypeSampledImage = 27;
constexpr uint32_t kOpTypeArray = 28;
constexpr uint32_t kOpTypeRuntimeArray = 29;
constexpr uint32_t kOpTypeStruct = 30;
constexpr uint32_t kOpTypePointer = 32;
constexpr uint32_t kOpConstant = 43;
constexpr uint32_t kOpVariable = 59;

// Decoration enum values
constexpr uint32_t kDecorationBlock = 2;
constexpr uint32_t kDecorationBufferBlock = 3;
constexpr uint32_t kDecorationLocation = 30;
constexpr uint32_t kDecorationBinding = 33;
constexpr uint32_t kDecorationDescriptorSet = 34;
constexpr uint32_t kDecorationOffset = 35;

// Storage class enum values
constexpr uint32_t kStorageClassUniformConstant = 0;
constexpr uint32_t kStorageClassInput = 1;
constexpr uint32_t kStorageClassUniform = 2;
constexpr uint32_t kStorageClassPushConstant = 9;
constexpr uint32_t kStorageClassStorageBuffer = 12;

// Image Dim enum values (subset)
// constexpr uint32_t kDim1D = 0;
// constexpr uint32_t kDim2D = 1;
// constexpr uint32_t kDim3D = 2;
// constexpr uint32_t kDimCube = 3;
constexpr uint32_t kDimBuffer = 5;

// Internal type records — populated as we encounter type-defining opcodes.
// Forward-reference safe in the sense that types must be defined before
// they're used (SPIR-V validation rule); we never have to defer lookups.
struct TypeInfo {
  enum class Kind {
    kUnknown,
    kInt,
    kFloat,
    kVector,
    kMatrix,
    kImage,
    kSampler,
    kSampledImage,
    kArray,
    kRuntimeArray,
    kStruct,
    kPointer,
  };
  Kind kind = Kind::kUnknown;

  // Common fields used selectively per kind. Not a union — wastes a few
  // bytes per type record, vastly simpler than tagged-union juggling.
  uint32_t inner = 0; // for kVector/kMatrix/kArray/kSampledImage/kPointer: element/wrapped type id
  uint32_t arrayLength = 1; // for kArray: number of elements (resolved from OpConstant). kRuntimeArray ⇒ 0.
  uint32_t storageClass = 0; // for kPointer
  uint32_t imageDim = 0; // for kImage
  uint32_t imageSampled = 0; // for kImage: 1 = sampled, 2 = storage. 0 = "either" (we treat as sampled).

  // for kStruct: cumulative member size by walking Offset decorations +
  // last member's size. Populated lazily from OpMemberDecorate Offset.
  std::vector<uint32_t> memberOffsets;

  // Width in bits, for kInt/kFloat.
  uint32_t width = 0;
};

struct VariableInfo {
  uint32_t typeId = 0; // OpTypePointer id
  uint32_t storageClass = 0;
};

// Decode the descriptorType for a UniformConstant variable from its
// pointee. SPIR-V (and shaderc-emitted GLSL) wraps things like:
//   sampler2D       ⇒ OpTypePointer UniformConstant → OpTypeSampledImage → OpTypeImage(Sampled=1)
//   image2D         ⇒ OpTypePointer UniformConstant → OpTypeImage(Sampled=2)
//   samplerBuffer   ⇒ OpTypePointer UniformConstant → OpTypeSampledImage → OpTypeImage(Dim=Buffer, Sampled=1)
//   imageBuffer     ⇒ OpTypePointer UniformConstant → OpTypeImage(Dim=Buffer, Sampled=2)
//   sampler         ⇒ OpTypePointer UniformConstant → OpTypeSampler
vk::DescriptorType descriptorTypeForUniformConstant(
  const TypeInfo& pointee,
  const std::map<uint32_t, TypeInfo>& types) {

  // Unwrap array if present (e.g. sampler2D arr[4]).
  const TypeInfo* t = &pointee;
  if (t->kind == TypeInfo::Kind::kArray || t->kind == TypeInfo::Kind::kRuntimeArray) {
    auto it = types.find(t->inner);
    if (it != types.end())
      t = &it->second;
  }

  if (t->kind == TypeInfo::Kind::kSampledImage) {
    // The wrapped image may be a buffer (samplerBuffer) → uniform texel buffer.
    auto imgIt = types.find(t->inner);
    if (imgIt != types.end() && imgIt->second.kind == TypeInfo::Kind::kImage && imgIt->second.imageDim == kDimBuffer) {
      return vk::DescriptorType::eUniformTexelBuffer;
    }
    return vk::DescriptorType::eCombinedImageSampler;
  }
  if (t->kind == TypeInfo::Kind::kImage) {
    if (t->imageDim == kDimBuffer) {
      // imageBuffer (storage texel) vs samplerBuffer (handled above as SampledImage)
      return vk::DescriptorType::eStorageTexelBuffer;
    }
    if (t->imageSampled == 2)
      return vk::DescriptorType::eStorageImage;
    return vk::DescriptorType::eSampledImage;
  }
  if (t->kind == TypeInfo::Kind::kSampler) {
    return vk::DescriptorType::eSampler;
  }
  // Anything else under UniformConstant: treat conservatively as a sampled
  // image — wrong choice will surface as a validation error.
  return vk::DescriptorType::eSampledImage;
}

uint32_t sizeOfType(const TypeInfo& t, const std::map<uint32_t, TypeInfo>& types) {
  switch (t.kind) {
    case TypeInfo::Kind::kInt:
    case TypeInfo::Kind::kFloat:
      return t.width / 8;
    case TypeInfo::Kind::kVector: {
      auto it = types.find(t.inner);
      if (it == types.end()) return 0;
      return sizeOfType(it->second, types) * t.arrayLength; // arrayLength reused as component count for vectors
    }
    case TypeInfo::Kind::kMatrix: {
      auto it = types.find(t.inner); // inner = column vector type
      if (it == types.end()) return 0;
      return sizeOfType(it->second, types) * t.arrayLength;
    }
    case TypeInfo::Kind::kArray: {
      auto it = types.find(t.inner);
      if (it == types.end()) return 0;
      // Without ArrayStride decoration we can't be precise; this is only
      // used as a fallback for the last-member case in push constants.
      return sizeOfType(it->second, types) * t.arrayLength;
    }
    case TypeInfo::Kind::kStruct: {
      if (t.memberOffsets.empty()) return 0;
      // Caller should have resolved an explicit total size; we don't have
      // per-member size info here. Push-constant size is reported as the
      // last member offset + that member's size; computing the size of an
      // arbitrary struct is out of scope.
      return t.memberOffsets.back();
    }
    default:
      return 0;
  }
}

} // namespace

SpirvReflection reflectSpirv(const std::vector<uint32_t>& spirv) {
  SpirvReflection out;
  if (spirv.size() < 5) {
    fprintf(stderr, "reflectSpirv: SPIR-V too short (%zu words)\n", spirv.size());
    abort();
  }
  if (spirv[0] != kSpirvMagic) {
    fprintf(stderr, "reflectSpirv: bad SPIR-V magic 0x%08x\n", spirv[0]);
    abort();
  }
  // const uint32_t version = spirv[1];
  // const uint32_t generator = spirv[2];
  const uint32_t idBound = spirv[3];
  (void) idBound;
  // word 4 = reserved (always 0).

  std::map<uint32_t, std::string> names; // <id> → debug name
  std::map<uint32_t, TypeInfo> types; // <id> → type info
  std::map<uint32_t, VariableInfo> variables; // <id> → var info
  std::map<uint32_t, uint32_t> constants; // <id> → uint constant value (for array length resolution)

  // Decoration tables keyed by target <id>.
  std::map<uint32_t, uint32_t> decoSet;
  std::map<uint32_t, uint32_t> decoBinding;
  std::map<uint32_t, uint32_t> decoLocation;
  std::map<uint32_t, bool> decoBlock;
  std::map<uint32_t, bool> decoBufferBlock;
  // For push constants we need the *last* member's offset to compute size.
  std::map<uint32_t, std::map<uint32_t, uint32_t>> memberOffsetByTarget;

  size_t cursor = 5;
  while (cursor < spirv.size()) {
    uint32_t word0 = spirv[cursor];
    uint32_t opcode = word0 & 0xFFFFu;
    uint32_t length = word0 >> 16;
    if (length == 0 || cursor + length > spirv.size()) {
      fprintf(stderr, "reflectSpirv: malformed instruction at word %zu (length=%u)\n", cursor, length);
      abort();
    }
    const uint32_t* operands = &spirv[cursor + 1];

    switch (opcode) {
      case kOpName: {
        // operands: <target-id> <literal-string-words>
        if (length >= 3) {
          uint32_t target = operands[0];
          const char* str = reinterpret_cast<const char*>(&operands[1]);
          names[target] = std::string(str);
        }
        break;
      }
      case kOpMemberName:
        // ignored — block members aren't named at the descriptor level
        break;

      case kOpDecorate: {
        // operands: <target-id> <decoration> [literal*]
        if (length < 3) break;
        uint32_t target = operands[0];
        uint32_t decoration = operands[1];
        switch (decoration) {
          case kDecorationDescriptorSet:
            if (length >= 4) decoSet[target] = operands[2];
            break;
          case kDecorationBinding:
            if (length >= 4) decoBinding[target] = operands[2];
            break;
          case kDecorationLocation:
            if (length >= 4) decoLocation[target] = operands[2];
            break;
          case kDecorationBlock:
            decoBlock[target] = true;
            break;
          case kDecorationBufferBlock:
            decoBufferBlock[target] = true;
            break;
        }
        break;
      }

      case kOpMemberDecorate: {
        // operands: <struct-id> <member-index> <decoration> [literal*]
        if (length < 4) break;
        uint32_t target = operands[0];
        uint32_t member = operands[1];
        uint32_t decoration = operands[2];
        if (decoration == kDecorationOffset && length >= 5) {
          memberOffsetByTarget[target][member] = operands[3];
        }
        break;
      }

      case kOpTypeInt: {
        if (length >= 4) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kInt;
          t.width = operands[1];
          types[id] = t;
        }
        break;
      }
      case kOpTypeFloat: {
        if (length >= 3) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kFloat;
          t.width = operands[1];
          types[id] = t;
        }
        break;
      }
      case kOpTypeVector: {
        if (length >= 4) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kVector;
          t.inner = operands[1];
          t.arrayLength = operands[2]; // component count
          types[id] = t;
        }
        break;
      }
      case kOpTypeMatrix: {
        if (length >= 4) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kMatrix;
          t.inner = operands[1]; // column type (vector)
          t.arrayLength = operands[2]; // column count
          types[id] = t;
        }
        break;
      }
      case kOpTypeImage: {
        // <id> <sampled-type-id> <dim> <depth> <arrayed> <ms> <sampled> <fmt> [<qualifier>]
        if (length >= 9) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kImage;
          t.imageDim = operands[2];
          t.imageSampled = operands[6];
          types[id] = t;
        }
        break;
      }
      case kOpTypeSampler: {
        if (length >= 2) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kSampler;
          types[id] = t;
        }
        break;
      }
      case kOpTypeSampledImage: {
        if (length >= 3) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kSampledImage;
          t.inner = operands[1];
          types[id] = t;
        }
        break;
      }
      case kOpTypeArray: {
        // <id> <element-type-id> <length-constant-id>
        if (length >= 4) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kArray;
          t.inner = operands[1];
          auto cit = constants.find(operands[2]);
          t.arrayLength = (cit != constants.end()) ? cit->second : 1;
          types[id] = t;
        }
        break;
      }
      case kOpTypeRuntimeArray: {
        if (length >= 3) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kRuntimeArray;
          t.inner = operands[1];
          t.arrayLength = 0;
          types[id] = t;
        }
        break;
      }
      case kOpTypeStruct: {
        if (length >= 2) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kStruct;
          // member ids are operands[1..length-2]; we don't store them here,
          // member-offset is folded in later via memberOffsetByTarget.
          types[id] = t;
        }
        break;
      }
      case kOpTypePointer: {
        // <id> <storage-class> <pointed-type-id>
        if (length >= 4) {
          uint32_t id = operands[0];
          TypeInfo t;
          t.kind = TypeInfo::Kind::kPointer;
          t.storageClass = operands[1];
          t.inner = operands[2];
          types[id] = t;
        }
        break;
      }

      case kOpConstant: {
        // <result-type> <result-id> <value-literal...>
        // We only need 32-bit unsigned/integer constants for array lengths.
        if (length >= 4) {
          uint32_t resultId = operands[1];
          constants[resultId] = operands[2];
        }
        break;
      }

      case kOpVariable: {
        // <result-type> <result-id> <storage-class> [<initializer>]
        if (length >= 4) {
          VariableInfo v;
          v.typeId = operands[0];
          v.storageClass = operands[2];
          variables[operands[1]] = v;
        }
        break;
      }

      default:
        // Unknown opcode — skip via length field.
        break;
    }

    cursor += length;
  }

  // Fold accumulated member-offset decorations into the struct type records
  // (push-constant size calculation needs them).
  for (auto& [structId, offsetMap] : memberOffsetByTarget) {
    auto tit = types.find(structId);
    if (tit == types.end()) continue;
    TypeInfo& ti = tit->second;
    if (ti.kind != TypeInfo::Kind::kStruct) continue;
    uint32_t maxMember = 0;
    for (const auto& [mi, _] : offsetMap)
      if (mi > maxMember) maxMember = mi;
    ti.memberOffsets.assign(maxMember + 1, 0);
    for (const auto& [mi, off] : offsetMap)
      ti.memberOffsets[mi] = off;
  }

  // Walk variables → emit descriptor bindings + vertex inputs.
  for (const auto& [varId, var] : variables) {
    auto ptrIt = types.find(var.typeId);
    if (ptrIt == types.end() || ptrIt->second.kind != TypeInfo::Kind::kPointer)
      continue;
    const TypeInfo& ptr = ptrIt->second;
    auto pointeeIt = types.find(ptr.inner);
    if (pointeeIt == types.end()) continue;
    const TypeInfo& pointee = pointeeIt->second;

    // Resolve the descriptor's display name: prefer the variable's OpName;
    // if empty (common for blocks declared without an instance name), fall
    // back to the pointee's OpName (the block-type name).
    auto getName = [&](uint32_t id) -> std::string {
      auto nit = names.find(id);
      if (nit != names.end() && !nit->second.empty()) return nit->second;
      return std::string();
    };

    switch (var.storageClass) {
      case kStorageClassUniformConstant: {
        SpirvDescriptorBinding b;
        b.name = getName(varId);
        if (b.name.empty()) b.name = getName(ptr.inner);
        auto sit = decoSet.find(varId);
        auto bit = decoBinding.find(varId);
        if (sit == decoSet.end() || bit == decoBinding.end()) break;
        b.set = sit->second;
        b.binding = bit->second;
        b.descriptorType = descriptorTypeForUniformConstant(pointee, types);
        // Array size: only counted if the immediate pointee is an array.
        if (pointee.kind == TypeInfo::Kind::kArray)
          b.arraySize = pointee.arrayLength;
        out.bindings.push_back(b);
        break;
      }

      case kStorageClassUniform: {
        // Could be a uniform buffer (Block) or storage buffer (BufferBlock,
        // the SPIR-V <1.3 way of declaring an SSBO).
        SpirvDescriptorBinding b;
        b.name = getName(varId);
        if (b.name.empty()) b.name = getName(ptr.inner);
        auto sit = decoSet.find(varId);
        auto bit = decoBinding.find(varId);
        if (sit == decoSet.end() || bit == decoBinding.end()) break;
        b.set = sit->second;
        b.binding = bit->second;
        if (decoBufferBlock.count(ptr.inner))
          b.descriptorType = vk::DescriptorType::eStorageBuffer;
        else
          b.descriptorType = vk::DescriptorType::eUniformBuffer;
        out.bindings.push_back(b);
        break;
      }

      case kStorageClassStorageBuffer: {
        // SPIR-V ≥1.3 SSBO path.
        SpirvDescriptorBinding b;
        b.name = getName(varId);
        if (b.name.empty()) b.name = getName(ptr.inner);
        auto sit = decoSet.find(varId);
        auto bit = decoBinding.find(varId);
        if (sit == decoSet.end() || bit == decoBinding.end()) break;
        b.set = sit->second;
        b.binding = bit->second;
        b.descriptorType = vk::DescriptorType::eStorageBuffer;
        out.bindings.push_back(b);
        break;
      }

      case kStorageClassPushConstant: {
        // pointee is the push-constant struct; report its total size.
        if (pointee.kind == TypeInfo::Kind::kStruct) {
          // Without per-member size info we can't be precise; report the
          // last member's offset as a lower bound. Good enough for the
          // engine's current shaders, which use std140 layouts with the
          // last member being a vec4 / mat4 chunk.
          uint32_t lastOffset = pointee.memberOffsets.empty() ? 0 : pointee.memberOffsets.back();
          // Add a conservative 16-byte tail (vec4) until we wire member-
          // type size resolution. TODO: when push constants actually get
          // used, replace with a real size walk via member-type ids.
          out.pushConstantSize = lastOffset + 16;
        }
        break;
      }

      case kStorageClassInput: {
        // Vertex inputs — only the vertex stage produces these usefully.
        auto lit = decoLocation.find(varId);
        if (lit == decoLocation.end()) break;
        SpirvVertexInput vi;
        vi.name = getName(varId);
        vi.location = lit->second;
        out.vertexInputs.push_back(vi);
        break;
      }

      default:
        break;
    }
  }

  // sizeOfType is referenced only for the struct fallback above; silence
  // -Wunused without forcing a separate translation unit.
  (void) sizeOfType;

  return out;
}
