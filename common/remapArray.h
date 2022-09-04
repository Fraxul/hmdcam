#pragma once
#include <cuda.h>
#include <opencv2/core/cuda.hpp>

void remapArray(CUtexObject src, cv::cuda::GpuMat xMap, cv::cuda::GpuMat yMap, cv::cuda::GpuMat dst, CUstream stream);

