#pragma once
#include "IArgusCamera.h"

class ArgusCameraMock : public IArgusCamera {
public:
  ArgusCameraMock(size_t sensorCount, unsigned int w, unsigned int h, double framerate);
  virtual ~ArgusCameraMock();

  // === ICameraProvider ===
  virtual size_t streamCount() const { return m_textures.size(); }
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const { return m_textures[sensorIndex]; }
  virtual CUtexObject cudaLumaTexObject(size_t sensorIndex) const { assert(false && "unimplemented"); return 0; }
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t sensorIdx);
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
  std::vector<RHISurface::ptr> m_textures;
  std::vector<VPIImage> m_vpiImages;
};


