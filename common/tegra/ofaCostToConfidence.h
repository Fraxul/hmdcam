#pragma once
#include <cuda.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>

void ofaCostToConfidence(CUtexObject ofaCost, cv::cuda::GpuMat& confidenceMat, uint8_t lowCostThreshold, uint8_t highCostThreshold, CUstream stream);

