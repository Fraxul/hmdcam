#pragma once
#include <opencv2/core/cuda.hpp>
#include <cuda.h>
#include "rhi/RHISurface.h"


namespace RHICUDA {

  void copyGpuMatToSurface(const cv::cuda::GpuMat&, RHISurface::ptr, const cv::cuda::Stream&);
  void copyGpuMatToSurface(const cv::cuda::GpuMat&, RHISurface::ptr, CUstream = 0);


  void copySurfaceToGpuMat(RHISurface::ptr, cv::cuda::GpuMat& gpuMat, const cv::cuda::Stream&);
  void copySurfaceToGpuMat(RHISurface::ptr, cv::cuda::GpuMat& gpuMat, CUstream = 0);

};

