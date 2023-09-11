#pragma once
#include <vector>
#include <epoxy/egl.h>

#include "common/ICameraProvider.h"
#include <cuda.h>
#include <cudaEGL.h>
#include <opencv2/core.hpp>

class IArgusCamera : public ICameraProvider {
public:
  IArgusCamera();
  virtual ~IArgusCamera();

  // === ICameraProvider (partial impl) ===
  virtual unsigned int streamWidth() const { return m_streamWidth; }
  virtual unsigned int streamHeight() const { return m_streamHeight; }

  // === IArgusCamera ===
  virtual size_t sessionCount() const = 0;
  virtual size_t sessionIndexForStream(size_t streamIdx) const = 0;
  virtual CUgraphicsResource cudaGraphicsResource(size_t sensorIndex) const = 0;
  virtual bool readFrame() = 0;
  virtual void stop() = 0;
  virtual void setRepeatCapture(bool) = 0;
  virtual void setExposureCompensation(float stops) = 0;

  virtual bool renderSettingsIMGUI() = 0;
  virtual bool renderPerformanceTuningIMGUI() = 0;
  virtual void loadSettings(cv::FileStorage&) = 0;
  virtual void saveSettings(cv::FileStorage&) = 0;

  // ====================


  // Misc simple accessors hoisted into base interface

  uint64_t targetCaptureIntervalNs() const { return m_targetCaptureIntervalNs; }
  void setTargetCaptureIntervalNs(uint64_t value) { m_targetCaptureIntervalNs = value; }

  bool didAdjustCaptureIntervalThisFrame() const { return m_didAdjustCaptureIntervalThisFrame; }
  bool willAdjustCaptureInterval() const { return m_adjustCaptureInterval; }
  void setAdjustCaptureInterval(bool value) { m_adjustCaptureInterval = value; }


  const glm::vec2& acRegionCenter() const { return m_acRegionCenter; }
  const glm::vec2& acRegionSize() const { return m_acRegionSize; }

  float exposureCompensation() const { return m_exposureCompensation; }

  struct FrameMetadata_t {
    uint64_t sensorTimestamp;
    uint64_t frameDurationNs;
    uint64_t sensorExposureTimeNs;
    uint32_t sensorSensitivityISO;
    float ispDigitalGain;
    float sensorAnalogGain;
  };

  // Metadata accessors for the current frame
  // Start timestamp for the sensor capture, in nanoseconds. Referenced to CLOCK_MONOTONIC.
  uint64_t frameSensorTimestamp(size_t sensorIndex) const { return m_frameMetadata[sensorIndex].sensorTimestamp; }
  const FrameMetadata_t& frameMetadata(size_t sensorIndex) const { return m_frameMetadata[sensorIndex]; }

protected:
  uint64_t m_targetCaptureIntervalNs;
  unsigned int m_streamWidth = 0, m_streamHeight = 0;
  float m_exposureCompensation = 0.0f;
  glm::vec2 m_acRegionCenter = glm::vec2(0.5f, 0.5f);
  glm::vec2 m_acRegionSize = glm::vec2(1.0f, 1.0f);

  // Per-stream per-frame metadata, populated for each frame in readFrame()
  std::vector<FrameMetadata_t> m_frameMetadata;

  bool m_adjustCaptureInterval = false;
  bool m_didAdjustCaptureIntervalThisFrame = false;
  int m_adjustCaptureCooldownFrames = 96;
  int m_adjustCaptureEvalWindowFrames = 64;
};

