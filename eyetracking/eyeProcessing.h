#pragma once
#include <cuda.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>

void ApplyLUT8to16(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const ushort* lut, CUstream stream);
void ComputeClassIndex(const void* classWeightsFP16, uchar* outClassIndex, uint32_t width, uint32_t height, CUstream stream);

