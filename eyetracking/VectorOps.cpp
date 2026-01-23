#include "VectorOps.h"
#include <assert.h>
#include <algorithm>

// Fast conversion of 0....255 range uint8 values to -1...1 range fp16 values
// Has a tiny precision loss around input 127 / output zero:
//
// Input  Expected fp16     Actual fp16
// ------------------------------------
//   124  -0.02745 a707   -0.02747 a708
//   125  -0.01961 a505   -0.01962 a506
//   126  -0.01176 a206   -0.01178 a208
//   127  -0.00392 9c04   -0.00394 9c08
//   128   0.00392 1c04    0.00391 1c00
//   129   0.01176 2206    0.01175 2204
//   130   0.01961 2505    0.01959 2504
//
// All other values are bitwise-identical to the reference algorithm, which converts/rescales in fp32 and then downcasts to fp16:
//     _Float16 f16 = static_cast<_Float16>((static_cast<float>(input_u8) / 127.5f) - 1.0f)
//
// (The accuracy of doing the convert/rescale in fp16 is worse than this version)
//
// Requires compiler ARM NEON FP16 support to be enabled: -march=armv8.2-a+fp16 (for both gcc and clang)
//
void convertUnorm8ToSnormFp16(const uint8_t* inU8, void* outFP16, const size_t elementCount) {
  // Only support chunks of 8 elements right now
  assert((elementCount & 7) == 0);

  float16x8_t* vectorOut = reinterpret_cast<float16x8_t*>(outFP16);
  const uint16x8_t signFixup = vdupq_n_u16(0x8000);
  for (size_t chunkIdx = 0; chunkIdx < (elementCount / 8); ++chunkIdx) {
    // Load 8x 8-bit values and widen to 16 bit
    uint16x8_t ux16 = vmovl_u8(vld1_u8(inU8 + (chunkIdx * 8)));

    // Replicate low 8 to high 8
    ux16 = vsliq_n_u16(ux16, ux16, 8);

    // Fixup sign bit for unorm to snorm conversion, then reinterpret as s16
    int16x8_t x = vreinterpretq_s16_u16(veorq_u16(ux16, signFixup));

    // Convert signed fixed-point to fp16
    float16x8_t f = vcvtq_n_f16_s16(x, 15);

    // Write output
    vectorOut[chunkIdx] = f;
  }
}

// Convert 0...255 uint8 values to DLA-int8 -127...127 range
// Conversion equation: out = max(in - 128, -127)
// The input value 0 would map to -128, but that is not valid for the DLA's narrow int8 range;
// input 0 is clipped to output -127.

void convertUnorm8ToDLAInt8(const uint8_t* inU8, void* outDLAInt8, size_t elementCount) {
  // Only support chunks of 16 elements right now
  assert((elementCount & 15) == 0);

  int8x16_t* vectorOut = reinterpret_cast<int8x16_t*>(outDLAInt8);
  const uint8x16_t signFixup = vdupq_n_u8(0x80);
  const int8x16_t cutoff = vdupq_n_s8(-127);

  for (size_t chunkIdx = 0; chunkIdx < (elementCount / 16); ++chunkIdx) {

    // Load 16x 8-bit values
    uint8x16_t ux = vld1q_u8(inU8 + (chunkIdx * 16));

    // Fixup sign bit for unorm to snorm conversion, then reinterpret as s8
    int8x16_t x = vreinterpretq_s8_u8(veorq_u8(ux, signFixup));

    // Max with cutoff value to remove out-of-bounds -128 value (DLA int8 is narrow, so -127...127)
    x = vmaxq_s8(x, cutoff);

    // Write output
    vectorOut[chunkIdx] = x;
  }
}

void fp16ThresholdToU8Mask(const _Float16* inFP16, _Float16 thresholdValue, uint8_t* outU8, size_t elementCount) {
  // Only support chunks of 8 elements right now
  assert((elementCount & 7) == 0);

  uint8x8_t* vectorOut = reinterpret_cast<uint8x8_t*>(outU8);
  const float16x8_t refval = vdupq_n_f16(thresholdValue);

  for (size_t chunkIdx = 0; chunkIdx < (elementCount / 8); ++chunkIdx) {
    // Load 8x fp16 values
    float16x8_t x = vld1q_f16(reinterpret_cast<const __fp16*>(inFP16 + (chunkIdx * 8)));

    // Compare to ref value. result is 16-bit 0000 (false) or ffff (true)
    uint16x8_t c16 = vcgeq_f16(x, refval);

    // Shift right and narrow to 8 bits
    uint8x8_t c8 = vshrn_n_u16(c16, 8);

    // TODO: May be able to compare two blocks of values, then reinterpret to u8 and use vuzp1_u8 to zip bytes of the compare results together for 16 elements at a time

    // Write output
    vectorOut[chunkIdx] = c8;
  }
}

// Returns largest element value
_Float16 fp16VectorMax(const _Float16* inFP16, size_t elementCount) {
  // Require at least one vector chunk.
  assert(elementCount >= 8);

  // Chunk size is 8 elements
  size_t chunks = elementCount / 8;
  size_t remainingElements = elementCount & 7;

  // Load initial value
  float16x8_t workVec = vld1q_f16(reinterpret_cast<const __fp16*>(inFP16));

  // Load and max subsequent values
  for (size_t chunkIdx = 1; chunkIdx < chunks; ++chunkIdx) {
    // Load 8x fp16 values
    float16x8_t x = vld1q_f16(reinterpret_cast<const __fp16*>(inFP16 + (chunkIdx * 8)));

    // Element-wise max
    workVec = vmaxq_f16(x, workVec);
  }

  // Pair-wise reducing max
  _Float16 result = vmaxnmvq_f16(workVec);

  // Process any remaining single elements at the end of the vector run
  if (remainingElements) {
    const _Float16* remainder = inFP16 + (chunks * 8);
    for (size_t p = 0; p < remainingElements; ++p) {
      result = std::max<_Float16>(result, remainder[p]);
    }
  }

  return result;
}

