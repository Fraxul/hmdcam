#pragma once
#include "IArgusCamera.h"
#include <vector>
#include <epoxy/egl.h>
#include <Argus/Argus.h>

#include "rhi/egl/RHIEGLImageSurfaceGL.h"
#include "rhi/RHIRenderTarget.h"
#include "common/ICameraProvider.h"
#include "common/ScrollingBuffer.h"
#include <cuda.h>
#include <cudaEGL.h>
#include <opencv2/core.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_count.hpp>

class ArgusCamera : public IArgusCamera {
public:
  ArgusCamera(EGLDisplay, EGLContext, double framerate);
  virtual ~ArgusCamera();

  // === ICameraProvider ===
  virtual size_t streamCount() const { return m_bufferPools.size(); }
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const { return m_bufferPools[sensorIndex].activeBuffer().rhiSurface; }
  virtual const char* rgbTextureGLSamplerType() const { return "samplerExternalOES"; }
  virtual CUtexObject cudaLumaTexObject(size_t sensorIndex) const { return m_bufferPools[sensorIndex].activeBuffer().cudaLumaTexObject; }
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t sensorIdx);
  virtual VPIImage vpiImage(size_t sensorIndex) const { return m_bufferPools[sensorIndex].activeBuffer().vpiImage; }
  // =======================

  // === IArgusCamera ===
  virtual CUgraphicsResource cudaGraphicsResource(size_t sensorIndex) const { return m_bufferPools[sensorIndex].activeBuffer().cudaResource; }
  virtual bool readFrame();
  virtual void stop();
  virtual void setRepeatCapture(bool);

  virtual void setExposureCompensation(float stops);

  // centerpoint and size are in normalized coordinates (0...1)
  // Default is the whole image: center=(0.5, 0.5), size = (1.0, 1.0).
  // Region will be clipped to the size of the image if it overhangs an edge (ex. if you move the center point without decreasing the size)
  void setAcRegion(const glm::vec2& center, const glm::vec2& size);

  virtual bool renderPerformanceTuningIMGUI();
  virtual bool renderSettingsIMGUI();
  virtual void loadSettings(cv::FileStorage&);
  virtual void saveSettings(cv::FileStorage&);

  void setCaptureDurationOffset(int64_t ns);
  int64_t captureDurationOffset() const;

  // ====================

  void populateGpuMat(size_t sensorIndex, cv::cuda::GpuMat&, const cv::cuda::Stream&);

private:
  void teardownCaptureSessions();
  void buildCaptureSessions();

  EGLDisplay m_display;
  EGLContext m_context;

  bool m_shouldResubmitCaptureRequest : 1;
  bool m_captureIsRepeating : 1;

  uint32_t m_failedCaptures = 0;

  uint32_t m_minAcRegionWidth, m_minAcRegionHeight;

  uint64_t m_currentCaptureDurationNs;
  uint64_t m_captureDurationMinNs, m_captureDurationMaxNs; // from sensor mode
  uint64_t m_previousSensorTimestampNs;
  unsigned int m_samplesAtCurrentDuration;

  typedef boost::accumulators::accumulator_set<double, boost::accumulators::stats<
      boost::accumulators::tag::rolling_mean,
      boost::accumulators::tag::rolling_count
    > > CaptureIntervalStats_t;
  CaptureIntervalStats_t m_captureIntervalStats;

  void setCaptureDurationNs(uint64_t captureDurationNs);

  // Shared camera provider
  Argus::UniqueObj<Argus::CameraProvider> m_cameraProvider;

  // Per-sensor objects
  std::vector<Argus::CameraDevice*> m_cameraDevices;
  std::vector<Argus::OutputStream*> m_outputStreams;

  Argus::SensorMode* m_sensorMode = nullptr;


  struct BufferPool {
    struct Entry {
      Entry() : argusBuffer(NULL), nativeBuffer(-1), eglImage(NULL), cudaResource(NULL) {}
      Entry(Argus::Buffer* b_, int nb_, EGLImageKHR egl_, CUgraphicsResource cr_) : argusBuffer(b_), nativeBuffer(nb_), eglImage(egl_), cudaResource(cr_) {}

      Argus::Buffer* argusBuffer;
      int nativeBuffer;
      EGLImageKHR eglImage;
      RHIEGLImageSurfaceGL::ptr rhiSurface;

      CUgraphicsResource cudaResource;
      CUeglFrame eglFrame;
      CUtexObject cudaLumaTexObject = 0;

      VPIImage vpiImage = nullptr;
    };

    std::vector<Entry> buffers;
    Entry& activeBuffer() { return buffers[activeBufferIndex]; }
    const Entry& activeBuffer() const { return buffers[activeBufferIndex]; }

    size_t activeBufferIndex;
    void setActiveBufferIndex(Argus::Buffer* b) {
      for (size_t i = 0; i < buffers.size(); ++i) {
        if (buffers[i].argusBuffer == b) {
          activeBufferIndex = i;
          return;
        }
      }
      assert(false && "ArgusCamera::BufferPool::setActiveBufferIndex(): buffer not in pool");
    }

  };

  std::vector<BufferPool> m_bufferPools;

  // Which buffers need to be released to the stream next readFrame
  std::vector<Argus::Buffer*> m_releaseBuffers;

  // Sessions and per-session objects
  struct SessionData {
    Argus::CaptureSession* m_captureSession = nullptr;
    Argus::Request* m_captureRequest = nullptr;
    Argus::EventQueue* m_completionEventQueue = nullptr; // for EVENT_TYPE_CAPTURE_COMPLETE
    int64_t m_durationSkew_ns = 0;
  };
  std::vector<SessionData> m_perSessionData;


  uint32_t m_streamsPerSession = 2;
  virtual size_t sessionCount() const { return m_perSessionData.size(); }
  virtual size_t sessionIndexForStream(size_t streamIdx) const { return streamIdx / m_streamsPerSession; }

  // Inter-session timing data
  struct SessionTimingData {
    SessionTimingData() { memset(timestampDelta, 0, sizeof(float) * 4); }

    float timestampDelta[4]; // in milliseconds; relative to session 0. no data for session 0 (would always be 0): timestampDelta[0] is for session 1, etc.
  };
  ScrollingBuffer<SessionTimingData> m_sessionTimingData = ScrollingBuffer<SessionTimingData>(512);


  bool m_adjustSessionSkew = true;

  mutable RHISurface::ptr m_tmpBlitSurface;
  mutable RHIRenderTarget::ptr m_tmpBlitRT;

  // noncopyable
  ArgusCamera(const ArgusCamera&);
  ArgusCamera& operator=(const ArgusCamera&);
};

