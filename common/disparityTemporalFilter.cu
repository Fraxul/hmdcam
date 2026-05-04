#include "disparityFill.h"
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
  PtrStep<uint8_t> debugOut) {

  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (x >= currentFrameDisparity.cols || y >= currentFrameDisparity.rows)
    return;

  // Read current and previous samples.
  uint16_t currentSample = currentFrameDisparity.ptr(y)[x];
  uint16_t previousSample = previousFrameDisparity.ptr(y)[x];

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
  cv::cuda::GpuMat& outDisparity, CUstream stream, cv::cuda::GpuMat* debugMat) {

  dim3 block(32, 4);
  dim3 grid(
    cv::cuda::device::divUp(currentFrameDisparity.cols, block.x),
    cv::cuda::device::divUp(currentFrameDisparity.rows, block.y));

  disparityTemporalFilterKernel<<<grid, block, 0, stream>>>(maxValidDisparityRaw, stableDepthThreshold, defaultAlpha,
    PtrStepSz<const uint16_t>(currentFrameDisparity.rows, currentFrameDisparity.cols, (const uint16_t*) currentFrameDisparity.cudaPtr(), currentFrameDisparity.step),
    PtrStep<const uint16_t>((const uint16_t*) previousFrameDisparity.cudaPtr(), previousFrameDisparity.step),
    PtrStep<uint16_t>((uint16_t*) outDisparity.cudaPtr(), outDisparity.step),
    PtrStep<uint8_t>((uint8_t*) (debugMat ? debugMat->cudaPtr() : nullptr), debugMat ? debugMat->step : 0));
}
