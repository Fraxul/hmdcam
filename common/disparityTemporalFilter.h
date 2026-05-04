#pragma once
#include <cuda.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

// Filters currentFrameDisparity and previousFrameDisparity and writes to outDisparityMat.
// It's OK if outDisparityMat aliases one of the {current, previous} frame mats.
void disparityTemporalFilter(uint16_t maxValidDisparityRaw, uint16_t stableDepthThreshold, uint8_t defaultAlpha,
  cv::cuda::GpuMat& currentFrameDisparity, cv::cuda::GpuMat& previousFrameDisparity, cv::cuda::GpuMat& outDisparity,
  CUstream stream, cv::cuda::GpuMat* debugMat = nullptr);
