#pragma once
#include <stddef.h>
#include "rhi/RHISurface.h"
#include <cuda.h>
#include <opencv2/core/cuda.hpp>

struct VPIImageImpl;
typedef struct VPIImageImpl* VPIImage;

class ICameraProvider {
public:
  virtual ~ICameraProvider() {}

  virtual size_t streamCount() const = 0;
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const = 0;
  virtual CUtexObject cudaLumaTexObject(size_t sensorIndex) const = 0;
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t sensorIndex) = 0;
  virtual VPIImage vpiImage(size_t sensorIndex) const = 0;
  virtual unsigned int streamWidth() const = 0;
  virtual unsigned int streamHeight() const = 0;
};

class NullCameraProvider : public ICameraProvider {
public:
  NullCameraProvider(size_t streamCount_, unsigned int w_ = 1920, unsigned int h_ = 1080) : m_streamCount(streamCount_), m_streamWidth(w_), m_streamHeight(h_) {}
  virtual ~NullCameraProvider() {}

  virtual size_t streamCount() const { return m_streamCount; }
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const { return RHISurface::ptr(); }
  virtual CUtexObject cudaLumaTexObject(size_t sensorIndex) const { return 0; }
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t sensorIndex) { return cv::cuda::GpuMat(); }
  virtual VPIImage vpiImage(size_t sensorIndex) const { return nullptr; }
  virtual unsigned int streamWidth() const { return m_streamWidth; }
  virtual unsigned int streamHeight() const { return m_streamHeight; }
protected:
  size_t m_streamCount;
  unsigned int m_streamWidth, m_streamHeight;
};

