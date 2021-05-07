#pragma once
#include <vector>
#include <EGL/egl.h>
#include <Argus/Argus.h>
#include "rhi/egl/RHIEGLStreamSurfaceGL.h"
#include "rhi/RHIRenderTarget.h"
#include "common/ICameraProvider.h"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_count.hpp>

class ArgusCamera : public ICameraProvider {
public:
  ArgusCamera(EGLDisplay, EGLContext, double framerate);
  virtual ~ArgusCamera();

  bool readFrame();

  void stop();

  virtual size_t streamCount() const { return m_eglStreams.size(); }
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const { return m_textures[sensorIndex]; }
  virtual void populateGpuMat(size_t sensorIndex, cv::cuda::GpuMat&, const cv::cuda::Stream&) const;
  virtual unsigned int streamWidth() const { return m_streamWidth; }
  virtual unsigned int streamHeight() const { return m_streamHeight; }

  // Metadata accessors for the current frame
  struct FrameMetadata_t {
    uint64_t sensorTimestamp;
    uint64_t sensorExposureTimeNs;
    uint32_t sensorSensitivityISO;
    float ispDigitalGain;
    float sensorAnalogGain;
  };
  // Start timestamp for the sensor capture, in nanoseconds. Referenced to CLOCK_MONOTONIC.
  uint64_t frameSensorTimestamp(size_t sensorIndex) const { return m_frameMetadata[sensorIndex].sensorTimestamp; }
  const FrameMetadata_t& frameMetadata(size_t sensorIndex) const { return m_frameMetadata[sensorIndex]; }


  uint64_t targetCaptureIntervalNs() const { return m_targetCaptureIntervalNs; }
  void setTargetCaptureIntervalNs(uint64_t value) { m_targetCaptureIntervalNs = value; }

  void setRepeatCapture(bool);

  bool didAdjustCaptureIntervalThisFrame() const { return m_didAdjustCaptureIntervalThisFrame; }
  bool willAdjustCaptureInterval() const { return m_adjustCaptureInterval; }
  void setAdjustCaptureInterval(bool value) { m_adjustCaptureInterval = value; }

private:
  EGLDisplay m_display;
  EGLContext m_context;

  std::vector<RHIEGLStreamSurfaceGL::ptr> m_textures;
  std::vector<EGLStreamKHR> m_eglStreams;
  unsigned int m_streamWidth, m_streamHeight;
  bool m_captureIsRepeating;

  uint64_t m_targetCaptureIntervalNs;

  uint64_t m_currentCaptureDurationNs;
  uint64_t m_captureDurationMinNs, m_captureDurationMaxNs; // from sensor mode
  uint64_t m_previousSensorTimestampNs;
  unsigned int m_samplesAtCurrentDuration;
  bool m_adjustCaptureInterval;
  bool m_didAdjustCaptureIntervalThisFrame;
  boost::accumulators::accumulator_set<double, boost::accumulators::stats<
      boost::accumulators::tag::rolling_mean,
      boost::accumulators::tag::rolling_count
    > > m_captureIntervalStats;

  void setCaptureDurationNs(uint64_t captureDurationNs);

  // Per-stream per-frame metadata, populated for each frame in readFrame()
  std::vector<FrameMetadata_t> m_frameMetadata;

  // Per-sensor objects
  std::vector<Argus::CameraDevice*> m_cameraDevices;
  std::vector<Argus::OutputStream*> m_outputStreams;

  // Session common objects
  Argus::UniqueObj<Argus::CameraProvider> m_cameraProvider;
  Argus::CaptureSession* m_captureSession;
  Argus::Request* m_captureRequest;
  Argus::EventQueue* m_completionEventQueue; // for EVENT_TYPE_CAPTURE_COMPLETE

  mutable RHISurface::ptr m_tmpBlitSurface;
  mutable RHIRenderTarget::ptr m_tmpBlitRT;

  // noncopyable
  ArgusCamera(const ArgusCamera&);
  ArgusCamera& operator=(const ArgusCamera&);
};

