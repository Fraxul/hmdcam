#pragma once
#include <stddef.h>
#include "rhi/RHISurface.h"
#include <opencv2/core/cuda.hpp>

struct VPIImageImpl;
typedef struct VPIImageImpl* VPIImage;

class ICameraProvider {
public:
  virtual ~ICameraProvider() {}

  virtual size_t streamCount() const = 0;
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const = 0;
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t sensorIndex) = 0;
  virtual VPIImage vpiImage(size_t sensorIndex) const = 0;
  virtual unsigned int streamWidth() const = 0;
  virtual unsigned int streamHeight() const = 0;
};

