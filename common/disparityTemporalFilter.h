#pragma once
#include <cuda.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

// Row-major 3x3 homography mapping a current-frame disparity pixel (x, y, 1) to the
// corresponding previous-frame disparity pixel, for IMU motion-compensated temporal
// filtering (so the blend integrates the same world point across frames rather than
// smearing edges during head rotation). Passing nullptr / identity disables reprojection
// and samples the previous frame at the same (x, y) -- the legacy behavior.
struct DisparityReprojection {
  float m[9];
};

// Filters currentFrameDisparity and previousFrameDisparity and writes to outDisparityMat.
// It's OK if outDisparityMat aliases one of the {current, previous} frame mats.
// previousFrameReprojection, if non-null, motion-compensates the previous-frame sample.
void disparityTemporalFilter(uint16_t maxValidDisparityRaw, uint16_t stableDepthThreshold, uint8_t defaultAlpha,
  cv::cuda::GpuMat& currentFrameDisparity, cv::cuda::GpuMat& previousFrameDisparity, cv::cuda::GpuMat& outDisparity,
  CUstream stream, const DisparityReprojection* previousFrameReprojection = nullptr, cv::cuda::GpuMat* debugMat = nullptr);
