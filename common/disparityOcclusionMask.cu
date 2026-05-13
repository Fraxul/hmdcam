#include "disparityOcclusionMask.h"
#include "rhi/cuda/CudaUtil.h"
#include <opencv2/core/cuda/common.hpp>

using namespace cv;
using namespace cv::cuda;

namespace {

__global__ void occlusionMaskKernel(
  PtrStepSz<uint16_t> disparityInOut,
  PtrStep<uint8_t> confidence,
  uint16_t maxValidDisparityRaw,
  float disparityPrescale,
  int searchWindowPixels,
  float hysteresisPixels,
  uint8_t confidenceCeiling,
  int smearLeftScanPixels) {

  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= disparityInOut.cols || py >= disparityInOut.rows)
    return;

  // If OFA was already confident in this match, trust it: a real right-view
  // match cannot also be in an occlusion shadow. Because this kernel never
  // writes to high-confidence pixels (here or below), the leftward smear
  // scan in occluded pixels can read these confidences and disparities
  // without a race.
  if (confidence.ptr(py)[px] >= confidenceCeiling)
    return;

  const uint16_t myDispRaw = disparityInOut.ptr(py)[px];
  // If this pixel's own disparity is already invalid, leave its confidence
  // alone (it was already going to be ignored by the splat).
  if (myDispRaw > maxValidDisparityRaw)
    return;

  const float myDispPixels = static_cast<float>(myDispRaw) * disparityPrescale;
  const int maxRightDx = min(searchWindowPixels, disparityInOut.cols - 1 - px);

  bool occluded = false;
  for (int dx = 1; dx <= maxRightDx; ++dx) {
    uint16_t neighborRaw = disparityInOut.ptr(py)[px + dx];
    if (neighborRaw > maxValidDisparityRaw)
      continue;
    float neighborPixels = static_cast<float>(neighborRaw) * disparityPrescale;
    float dispJumpPixels = neighborPixels - myDispPixels;
    // Foreground at (px+dx) shadows (px) iff its disparity exceeds the column
    // distance back to (px), plus a hysteresis slack to ignore disparity
    // noise on flat regions.
    if (dispJumpPixels >= static_cast<float>(dx) + hysteresisPixels) {
      occluded = true;
      break;
    }
  }

  if (!occluded)
    return;

  // Try to smear a background disparity in from the left: scan leftward for
  // the first high-confidence (>= confidenceCeiling) pixel with a valid
  // disparity and copy it. If found, the pixel keeps a partial confidence
  // (ceiling / 2) so the bilateral splat treats it as a soft inpainting
  // prior on the smeared value -- weaker than a real OFA match, but not
  // ignored. If smearing fails (or is disabled), confidence drops to zero
  // so the splat skips it entirely and the solver relies on smoothness to
  // fill the gap.
  //
  // Race-free w.r.t. the partial confidence: ceiling / 2 < ceiling, so a
  // smeared pixel is never accepted as a smear source by other threads
  // (the source check requires conf >= ceiling). High-confidence pixels
  // are never modified by this kernel (top-of-function early-out), so the
  // leftward scan reads stable source data.
  uint8_t newConf = 0;
  if (smearLeftScanPixels > 0) {
    const int maxLeftDx = min(smearLeftScanPixels, px);
    for (int dx = 1; dx <= maxLeftDx; ++dx) {
      int srcPx = px - dx;
      uint8_t srcConf = confidence.ptr(py)[srcPx];
      if (srcConf < confidenceCeiling)
        continue;
      uint16_t srcDisp = disparityInOut.ptr(py)[srcPx];
      if (srcDisp > maxValidDisparityRaw)
        continue; // high-conf but invalid -- keep scanning
      disparityInOut.ptr(py)[px] = srcDisp;
      newConf = confidenceCeiling / 2;
      break;
    }
  }

  confidence.ptr(py)[px] = newConf;
}

} // namespace

void disparityOcclusionMask(
  cv::cuda::GpuMat& disparityInOut,
  cv::cuda::GpuMat& confidence,
  uint16_t maxValidDisparityRaw,
  float disparityPrescale,
  uint32_t searchWindowPixels,
  float hysteresisPixels,
  uint8_t confidenceCeiling,
  uint32_t smearLeftScanPixels,
  CUstream stream) {

  CV_Assert(disparityInOut.type() == CV_16U);
  CV_Assert(confidence.type() == CV_8U);
  CV_Assert(disparityInOut.cols == confidence.cols && disparityInOut.rows == confidence.rows);

  if (searchWindowPixels == 0)
    return;

  dim3 block(32, 4);
  dim3 grid(
    cv::cuda::device::divUp(disparityInOut.cols, block.x),
    cv::cuda::device::divUp(disparityInOut.rows, block.y));

  occlusionMaskKernel<<<grid, block, 0, stream>>>(
    PtrStepSz<uint16_t>(disparityInOut.rows, disparityInOut.cols,
      (uint16_t*) disparityInOut.cudaPtr(), disparityInOut.step),
    PtrStep<uint8_t>((uint8_t*) confidence.cudaPtr(), confidence.step),
    maxValidDisparityRaw,
    disparityPrescale,
    static_cast<int>(searchWindowPixels),
    hysteresisPixels,
    confidenceCeiling,
    static_cast<int>(smearLeftScanPixels));
}
