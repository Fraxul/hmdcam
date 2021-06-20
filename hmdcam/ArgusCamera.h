#pragma once
#include <vector>
#include <EGL/egl.h>
#include <Argus/Argus.h>
#include "rhi/egl/RHIEGLImageSurfaceGL.h"
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

  virtual size_t streamCount() const { return m_textures.size(); }
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const { return m_textures[sensorIndex]; }
  virtual void populateGpuMat(size_t sensorIndex, cv::cuda::GpuMat&, const cv::cuda::Stream&);
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t sensorIdx);
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

  std::vector<RHIEGLImageSurfaceGL::ptr> m_textures;
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


  struct BufferPool {
    struct Entry {
      Entry() : argusBuffer(NULL), nativeBuffer(-1), eglImage(NULL), cudaResource(NULL) {}
      Entry(Argus::Buffer* b_, int nb_, EGLImageKHR egl_, CUgraphicsResource cr_) : argusBuffer(b_), nativeBuffer(nb_), eglImage(egl_), cudaResource(cr_) {}

      Argus::Buffer* argusBuffer;
      int nativeBuffer;
      EGLImageKHR eglImage;
      CUgraphicsResource cudaResource;
    };

    std::vector<Entry> buffers;
    Entry& activeBuffer() { return buffers[activeBufferIndex]; }

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

