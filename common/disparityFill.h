#pragma once
#include <cuda.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

void disparityFill(CUtexObject chromaTex, cv::cuda::GpuMat& disparityMat, float maxValidDisparityRaw, std::vector<cv::cuda::GpuMat>& disparityMinMaxMips, CUstream stream);

