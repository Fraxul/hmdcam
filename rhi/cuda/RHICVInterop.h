#pragma once
#include <opencv2/core/cuda.hpp>
#include <cuda.h>
#include "rhi/RHISurface.h"


namespace RHICUDA {

  void copyGpuMatToSurface(const cv::cuda::GpuMat&, RHISurface::ptr, CUstream stream = 0);

};

