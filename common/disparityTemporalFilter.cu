#include "disparityTemporalFilter.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/saturate_cast.hpp>
#include <opencv2/core/cuda/filters.hpp>

using namespace cv;
using namespace cv::cuda;

__global__ void disparityTemporalFilterKernel(uint16_t maxValidDisparityRaw, uint16_t stableDepthThreshold, uint8_t defaultAlpha,
  PtrStepSz<const uint16_t> currentFrameDisparity, PtrStep<const uint16_t> previousFrameDisparity,
  PtrStep<uint16_t> outDisparity,
  DisparityReprojection previousReprojection,
  PtrStep<uint8_t> debugOut) {

  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= currentFrameDisparity.cols || y >= currentFrameDisparity.rows)
    return;

  // Read current sample.
  uint16_t currentSample = currentFrameDisparity.ptr(y)[x];

  // Motion-compensate the previous-frame sample: map this pixel through the inter-frame
  // homography to where the same world point sat in the previous frame, then bilinearly
  // sample. Invalid taps (> maxValidDisparityRaw) are dropped from the interpolation and
  // the weights renormalized over the valid neighbors, so the invalid sentinel never bleeds
  // into a real disparity. An identity homography lands exactly on (x, y) and reduces to
  // the legacy single-tap read. A fully out-of-bounds / all-invalid neighborhood yields the
  // invalid sentinel (0xffff) so the blend falls back to the current sample.
  const float* h = previousReprojection.m;
  const float fxp = static_cast<float>(x);
  const float fyp = static_cast<float>(y);
  const float invW = 1.0f / ((h[6] * fxp) + (h[7] * fyp) + h[8]);
  const float sampleX = ((h[0] * fxp) + (h[1] * fyp) + h[2]) * invW;
  const float sampleY = ((h[3] * fxp) + (h[4] * fyp) + h[5]) * invW;

  const int x0 = static_cast<int>(floorf(sampleX));
  const int y0 = static_cast<int>(floorf(sampleY));
  const float ax = sampleX - static_cast<float>(x0);
  const float ay = sampleY - static_cast<float>(y0);

  float weightSum = 0.0f;
  float disparitySum = 0.0f;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < 2; ++i) {
      const int sx = x0 + i;
      const int sy = y0 + j;
      if (sx < 0 || sy < 0 || sx >= currentFrameDisparity.cols || sy >= currentFrameDisparity.rows)
        continue;
      const uint16_t tap = previousFrameDisparity.ptr(sy)[sx];
      if (tap > maxValidDisparityRaw)
        continue; // drop invalid taps so the sentinel doesn't pollute the average
      const float weight = (i ? ax : (1.0f - ax)) * (j ? ay : (1.0f - ay));
      weightSum += weight;
      disparitySum += weight * static_cast<float>(tap);
    }
  }
  const uint16_t previousSample = (weightSum > 0.0f)
    ? static_cast<uint16_t>(__float2uint_rn(disparitySum / weightSum))
    : 0xffffu;

  uint8_t outDebug;

  // Sample-blending decision tree
  uint16_t outSample;
  if (currentSample > maxValidDisparityRaw) {
    // Current sample is invalid, use previous sample.
    outSample = previousSample;
    outDebug = 0;
  } else if (previousSample > maxValidDisparityRaw) {
    // Previous sample is invalid, use current sample only.
    outSample = currentSample;
    outDebug = 255;
  } else {
    // Both samples are valid, see if we can blend them.
    int16_t delta = abs(static_cast<int16_t>(currentSample) - static_cast<int16_t>(previousSample));
    if (delta > stableDepthThreshold) {
      // Sample-delta is too large, this is probably a dynamic object. Use current sample only.
      outSample = currentSample;
      outDebug = 160;
    } else {
      // Blend samples.
      uint16_t alpha = defaultAlpha; // expand to u16 so (256 - alpha) works correctly
      outSample = (__umul24(alpha, currentSample) + __umul24(256u - alpha, previousSample)) >> 8;

      outDebug = 40;
    }
  }

  // Write in-place to currentFrameDisparity
  outDisparity.ptr(y)[x] = outSample;

  // Optional debug output
  if (debugOut.ptr()) {
    debugOut.ptr(y)[x] = outDebug;
  }
}

void disparityTemporalFilter(uint16_t maxValidDisparityRaw, uint16_t stableDepthThreshold, uint8_t defaultAlpha,
  cv::cuda::GpuMat& currentFrameDisparity, cv::cuda::GpuMat& previousFrameDisparity,
  cv::cuda::GpuMat& outDisparity, CUstream stream, const DisparityReprojection* previousFrameReprojection, cv::cuda::GpuMat* debugMat) {

  // Default to the identity homography (sample previous frame at the same pixel).
  DisparityReprojection reprojection = previousFrameReprojection
    ? *previousFrameReprojection
    : DisparityReprojection{
        {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}
  };

  dim3 block(32, 4);
  dim3 grid(
    cv::cuda::device::divUp(currentFrameDisparity.cols, block.x),
    cv::cuda::device::divUp(currentFrameDisparity.rows, block.y));

  disparityTemporalFilterKernel<<<grid, block, 0, stream>>>(maxValidDisparityRaw, stableDepthThreshold, defaultAlpha,
    PtrStepSz<const uint16_t>(currentFrameDisparity.rows, currentFrameDisparity.cols, (const uint16_t*) currentFrameDisparity.cudaPtr(), currentFrameDisparity.step),
    PtrStep<const uint16_t>((const uint16_t*) previousFrameDisparity.cudaPtr(), previousFrameDisparity.step),
    PtrStep<uint16_t>((uint16_t*) outDisparity.cudaPtr(), outDisparity.step),
    reprojection,
    PtrStep<uint8_t>((uint8_t*) (debugMat ? debugMat->cudaPtr() : nullptr), debugMat ? debugMat->step : 0));
}
