#pragma once
#include <cuda.h>
#include <opencv2/core/cuda.hpp>
#include <stdint.h>

// Pre-pass for stereo half-occlusion. For each pixel in the left-view
// disparity image, scans rightward up to searchWindowPixels columns; if any
// pixel in that window has a disparity high enough to geometrically shadow
// the current column (i.e. the foreground at (px+dx) is shifted left far
// enough in the right view to cover the right-view location of (px)), the
// current pixel is marked as occluded and its confidence is set to zero.
//
// Geometric test in pixel units: pixel at column px with disparity D_bg is
// occluded by a foreground pixel at column px+dx with disparity D_fg iff
//   D_fg - D_bg >= dx
// (the foreground "shadow" reaches back by D_fg - D_bg columns in the left
// view). A small hysteresis is added on the right-hand side so that small
// disparity noise around real edges does not trip the test.
//
// disparityPrescale converts the raw uint16 disparity values to pixel-units
// of horizontal shift (= 1 / (1 << subpixelFractionalBits) for the OFA
// backend). The bilateral solver / temporal filter downstream see the
// modified confidence and treat occluded vertices as gaps to inpaint from
// luma-matching neighbours.
//
// confidenceCeiling: pixels whose own OFA confidence is >= this value are
// presumed to have a real right-view match and are never marked occluded.
// This protects against false positives where the geometric heuristic
// would catch high-confidence pixels at sharp but real disparity edges.
//
// smearLeftScanPixels: if > 0, after marking a pixel occluded the kernel
// also tries to overwrite its disparity with the nearest high-confidence
// disparity to the left, within this scan window. When the smear succeeds,
// confidence is set to confidenceCeiling/2 (a soft inpainting prior: the
// smeared value contributes to the bilateral splat at half a real match's
// weight). When it fails (no high-confidence source found within the
// window), confidence drops to 0 and the splat ignores the pixel entirely.

void disparityOcclusionMask(
  cv::cuda::GpuMat& disparityInOut,
  cv::cuda::GpuMat& confidence,
  uint16_t maxValidDisparityRaw,
  float disparityPrescale,
  uint32_t searchWindowPixels,
  float hysteresisPixels,
  uint8_t confidenceCeiling,
  uint32_t smearLeftScanPixels,
  CUstream stream);
