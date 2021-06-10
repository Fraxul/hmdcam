#pragma once
#include <stddef.h>
#include "rhi/RHISurface.h"
#include <opencv2/core/cuda.hpp>

class ICameraProvider {
public:
  virtual ~ICameraProvider() {}

  virtual size_t streamCount() const = 0;
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const = 0;
  virtual void populateGpuMat(size_t sensorIndex, cv::cuda::GpuMat&, const cv::cuda::Stream& = cv::cuda::Stream::Null()) const = 0;
  virtual unsigned int streamWidth() const = 0;
  virtual unsigned int streamHeight() const = 0;
};

