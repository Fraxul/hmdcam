#pragma once
#include <cuda.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>

void remapArray(CUtexObject src, cv::Size inputImageSize, cv::cuda::GpuMat undistortRectifyMap, cv::cuda::GpuMat dst, CUstream stream);

// Builds a GpuMat suitable for feeding to remapArray. Parameters identical to cv::initUndistortRectifyMap
// Resultant GpuMat size is (inputImageSize / downsampleFactor)
cv::cuda::GpuMat remapArray_initUndistortRectifyMap(cv::InputArray cameraMatrix, cv::InputArray distCoeffs, cv::InputArray rectification, cv::InputArray newProjection, cv::Size inputImageSize, unsigned int downsampleFactor = 1);

