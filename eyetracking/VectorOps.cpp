#include "VectorOps.h"
#include <assert.h>
#include <algorithm>
#include <arm_acle.h>

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

// Threshold operation on fp16 values array
// Output uint8_t elements are 0 if input is < thresholdValue, 0xff if input is >= thresholdValue
// elementCount must be a multiple of 16.
void fp16ThresholdToU8Mask(const _Float16* inFP16, _Float16 thresholdValue, uint8_t* outU8, size_t elementCount) {
  // Only support chunks of 16 elements right now
  assert((elementCount & 15) == 0);

  uint8x16_t* vectorOut = reinterpret_cast<uint8x16_t*>(outU8);
  const float16x8_t refval = vdupq_n_f16(thresholdValue);

  for (size_t chunkIdx = 0; chunkIdx < (elementCount / 16); ++chunkIdx) {
    size_t baseIdx = chunkIdx * 16;

    // Load 16x fp16 values
    float16x8_t x0 = vld1q_f16(reinterpret_cast<const __fp16*>(inFP16 + baseIdx));
    float16x8_t x1 = vld1q_f16(reinterpret_cast<const __fp16*>(inFP16 + baseIdx + 8));

    // Compare to ref value. result is 16-bit 0000 (false) or ffff (true)
    uint16x8_t c16_0 = vcgeq_f16(x0, refval);
    uint16x8_t c16_1 = vcgeq_f16(x1, refval);

    // Narrow to 8-bit masks
    uint8x8_t c8_0 = vmovn_u16(c16_0);
    uint8x8_t c8_1 = vmovn_u16(c16_1);

    // Interleave the two 8-byte results into a single 16-byte result
    uint8x16_t c8 = vcombine_u8(c8_0, c8_1);

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

// Convert single-channel u8 input to rgba8 output, with alpha = 0xff
void convertGrayToRGBA(const uint8_t* inGray, uint8_t* outRGBA, size_t count) {
  // Broadcast 0xff into all 16 lanes of a 128-bit vector — this is our alpha channel
  const uint8x16_t alpha = vdupq_n_u8(0xff);
  size_t vectorCount = count & ~size_t(15);
  for (size_t i = 0; i < vectorCount; i += 16) {
    // Load 16 contiguous gray values into a 128-bit NEON register
    uint8x16_t g = vld1q_u8(inGray + i);

    // Pack the R, G, B, A channels as a struct-of-vectors:
    //   .val[0] = R = g   (gray copied to red)
    //   .val[1] = G = g   (gray copied to green)
    //   .val[2] = B = g   (gray copied to blue)
    //   .val[3] = A = 0xff
    uint8x16x4_t rgba = {{g, g, g, alpha}};

    // Store with 4-lane interleave: takes the struct-of-vectors layout above
    // and writes 64 bytes in array-of-structs (R,G,B,A,R,G,B,A,...) order.
    // The hardware handles the de-interleave, so we get 16 RGBA pixels in one store.
    vst4q_u8(outRGBA + i * 4, rgba);
  }
  // Scalar fallback for any trailing pixels beyond the last full 16-element vector
  for (size_t i = vectorCount; i < count; ++i) {
    uint8_t c = inGray[i];
    outRGBA[i * 4 + 0] = c;
    outRGBA[i * 4 + 1] = c;
    outRGBA[i * 4 + 2] = c;
    outRGBA[i * 4 + 3] = 0xff;
  }
}

// 6x6 area downsample of an input image using a box filter:
// Output image dimensions is (inputWidth/6, inputHeight/6)
// Each output pixel is the average of the corresponding 6x6 pixel area in the input image.
void areaDownsample6x6(const uint8_t* __restrict inU8, size_t inputRowStride, uint8_t* __restrict outU8, size_t outputWidth, size_t outputHeight, size_t outputRowStride) {
  // Output line-width is guaranteed to be a multiple of 8 pixels wide.
  assert((outputWidth & 7) == 0);

  // Division by 36 via vqrdmulh: sat_s16((2*a*910 + 2^15) >> 16) ~= round(a/36)
  const int16x8_t kDiv36 = vdupq_n_s16(910);

  // Row-streaming approach: process one input row at a time left-to-right across the full
  // output width. This gives the HW prefetcher a single sequential DRAM stream (optimal for
  // bandwidth), with partial sums kept in a tiny L1-resident buffer.
  // The output width is generally small, so we just keep it on the stack.
  alignas(16) uint16_t partialBuf[outputWidth];

#define preload(x) __pldx(0, 0, 1, x)
  for (size_t outputY = 0; outputY < outputHeight; ++outputY) {
    uint8_t* outputRow = outU8 + (outputY * outputRowStride);
    const uint8_t* baseRow = inU8 + ((outputY * 6) * inputRowStride);

    // Row 0: initialize partial sums
    {
      const uint8_t* row = baseRow;
      preload(row); preload(row + 64); preload(row + 128);
      for (size_t outputX = 0, off = 0; outputX < outputWidth; outputX += 8, off += 48) {
        uint8x16x3_t x = vld3q_u8(row + off);
        uint16x8_t s = vpaddlq_u8(x.val[0]);
        s = vpadalq_u8(s, x.val[1]);
        s = vpadalq_u8(s, x.val[2]);
        vst1q_u16(partialBuf + outputX, s);
      }
    }

    // Rows 1-4: accumulate into partial sums (one row at a time for single-stream DRAM access)
    for (size_t rowIdx = 1; rowIdx < 5; ++rowIdx) {
      const uint8_t* row = baseRow + rowIdx * inputRowStride;
      preload(row); preload(row + 64); preload(row + 128);
      for (size_t outputX = 0, off = 0; outputX < outputWidth; outputX += 8, off += 48) {
        uint16x8_t s = vld1q_u16(partialBuf + outputX);
        uint8x16x3_t x = vld3q_u8(row + off);
        s = vpadalq_u8(s, x.val[0]);
        s = vpadalq_u8(s, x.val[1]);
        s = vpadalq_u8(s, x.val[2]);
        vst1q_u16(partialBuf + outputX, s);
      }
    }

    // Row 5 + divide: merge last accumulation with division
    {
      const uint8_t* row = baseRow + 5 * inputRowStride;
      preload(row); preload(row + 64); preload(row + 128);
      for (size_t outputX = 0, off = 0; outputX < outputWidth; outputX += 8, off += 48) {
        uint16x8_t s = vld1q_u16(partialBuf + outputX);
        uint8x16x3_t x = vld3q_u8(row + off);
        s = vpadalq_u8(s, x.val[0]);
        s = vpadalq_u8(s, x.val[1]);
        s = vpadalq_u8(s, x.val[2]);
        int16x8_t div = vqrdmulhq_s16(vreinterpretq_s16_u16(s), kDiv36);
        uint8x8_t res = vqmovun_s16(div);
        vst1_u8(outputRow + outputX, res);
      }
    }
  }
#undef preload
}

