#pragma once
#include <cuda.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>

void medianFilter3x3_u16(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, CUstream stream);

