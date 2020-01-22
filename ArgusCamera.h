#pragma once
#include <vector>
#include <EGL/egl.h>
#include <Argus/Argus.h>
#include "rhi/gl/RHIEGLStreamSurfaceGL.h"

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_count.hpp>

class ArgusCamera {
public:
  ArgusCamera(EGLDisplay, EGLContext, std::vector<unsigned int> cameraIndices, unsigned int width, unsigned int height, double framerate);
  ~ArgusCamera();

  bool readFrame();

  void stop();

  size_t streamCount() const { return m_eglStreams.size(); }
  RHISurface::ptr rgbTexture(size_t sensorIndex) const { return m_textures[sensorIndex]; }
  // Start timestamp for the sensor capture, in nanoseconds. Appears to be referenced to CLOCK_MONOTONIC, though that's not documented...
  uint64_t sensorTimestamp(size_t sensorIndex) const { return m_sensorTimestamps[sensorIndex]; }
  unsigned int streamWidth() const { return m_streamWidth; }
  unsigned int streamHeight() const { return m_streamHeight; }

  uint64_t targetCaptureIntervalNs() const { return m_targetCaptureIntervalNs; }
  void setTargetCaptureIntervalNs(uint64_t value) { m_targetCaptureIntervalNs = value; }

  void setRepeatCapture(bool);

private:
  EGLDisplay m_display;
  EGLContext m_context;

  std::vector<unsigned int> m_cameraIds;
  std::vector<RHIEGLStreamSurfaceGL::ptr> m_textures;
  std::vector<EGLStreamKHR> m_eglStreams;
  unsigned int m_streamWidth, m_streamHeight;
  bool m_captureIsRepeating;

  uint64_t m_targetCaptureIntervalNs;

  uint64_t m_currentCaptureDurationNs;
  uint64_t m_captureDurationMinNs, m_captureDurationMaxNs; // from sensor mode
  uint64_t m_previousSensorTimestampNs;
  unsigned int m_samplesAtCurrentDuration;
  boost::accumulators::accumulator_set<double, boost::accumulators::stats<
      boost::accumulators::tag::rolling_mean,
      boost::accumulators::tag::rolling_count
    > > m_captureIntervalStats;

  void setCaptureDurationNs(uint64_t captureDurationNs);

  // Per-stream metadata, populated for each frame in readFrame()
  std::vector<uint64_t> m_sensorTimestamps;

  // Per-sensor objects
  std::vector<Argus::CameraDevice*> m_cameraDevices;
  std::vector<Argus::OutputStream*> m_outputStreams;

  // Session common objects
  Argus::UniqueObj<Argus::CameraProvider> m_cameraProvider;
  Argus::CaptureSession* m_captureSession;
  Argus::Request* m_captureRequest;
  Argus::EventQueue* m_completionEventQueue; // for EVENT_TYPE_CAPTURE_COMPLETE

  // noncopyable
  ArgusCamera(const ArgusCamera&);
  ArgusCamera& operator=(const ArgusCamera&);
};

