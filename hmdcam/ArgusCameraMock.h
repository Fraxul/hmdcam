#pragma once
#include "IArgusCamera.h"

class ArgusCameraMock : public IArgusCamera {
public:
  ArgusCameraMock(size_t sensorCount, unsigned int w, unsigned int h, double framerate);
  virtual ~ArgusCameraMock();

  // === ICameraProvider ===
  virtual size_t streamCount() const { return m_streamData.size(); }
  virtual RHISurface::ptr rgbTexture(size_t sensorIdx) const { return m_streamData[sensorIdx].rgbTexture; }
  virtual const char* rgbTextureGLSamplerType() const { return "sampler2D"; }
  virtual CUtexObject cudaLumaTexObject(size_t sensorIdx) const { return m_streamData[sensorIdx].cudaLumaTexObject; }
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t sensorIdx) { return m_streamData[sensorIdx].lumaGpuMat; }
  virtual VPIImage vpiImage(size_t sensorIndex) const;

  // === IArgusCamera ===
  virtual size_t sessionCount() const { return 1; }
  virtual size_t sessionIndexForStream(size_t streamIdx) const { return 0; }

  virtual CUgraphicsResource cudaGraphicsResource(size_t sensorIndex) const { return nullptr; }
  virtual bool readFrame();

  virtual void stop() {}
  virtual void setRepeatCapture(bool) {}
  virtual void setExposureCompensation(float stops) {}
  virtual bool renderPerformanceTuningIMGUI() { return false; }
  virtual bool renderSettingsIMGUI() { return false; }
  virtual void loadSettings(cv::FileStorage&) {}
  virtual void saveSettings(cv::FileStorage&) {}

protected:
  uint64_t m_previousFrameReadTime = 0;
  struct Stream {
    RHISurface::ptr rgbTexture;
    VPIImage vpiImage;
    cv::cuda::GpuMat lumaGpuMat;
    CUtexObject cudaLumaTexObject;
  };
  std::vector<Stream> m_streamData;
};


