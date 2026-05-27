#pragma once
#include "IArgusCamera.h"
#include <vector>
#include <epoxy/egl.h>
#include <Argus/Argus.h>

#include "rhi/RHIRenderTarget.h"
#include "rhi/RHISurface.h"
#include "rhi/vk/RHISamplerVK.h"
#include "hmdcam/tegra/ArgusCameraSurfaceVK.h"
#include "common/ICameraProvider.h"
#include "common/ScrollingBuffer.h"
#include <cuda.h>
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
  virtual size_t streamCount() const { return m_perSensorData.size(); }
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const { return m_perSensorData[sensorIndex].m_bufferPool.activeBuffer().rhiSurface; }
  virtual const char* rgbTextureGLSamplerType() const { return "sampler2D"; }
  virtual CUtexObject cudaLumaTexObject(size_t sensorIndex) const { return m_perSensorData[sensorIndex].m_bufferPool.activeBuffer().rhiSurface->cudaLumaTexObject(); }
  virtual CUtexObject cudaChromaTexObject(size_t sensorIndex) const { return m_perSensorData[sensorIndex].m_bufferPool.activeBuffer().rhiSurface->cudaChromaTexObject(); }
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t sensorIdx);
  virtual bool isStreamFailed(size_t sensorIndex) const;
  virtual RHISampler::ptr cameraSampler() const override { return m_cameraSampler; }
  // =======================

  // === IArgusCamera ===
  virtual bool fillCudaMemcpy2DForStreamSource(CUDA_MEMCPY2D& outCopyDescriptor, size_t sensorIndex, bool fromChromaPlane) const;
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

  virtual void setUsingExternalSync(bool usingExternalSync) {
    m_usingExternalSync = usingExternalSync;

    if (usingExternalSync) {
      // Turn off session skew and capture interval adjustments in extsync mode, since we don't control the timing.
      m_adjustSessionSkew = false;
      m_adjustCaptureInterval = false;
    }
  }

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

  typedef boost::accumulators::accumulator_set<double, boost::accumulators::stats<boost::accumulators::tag::rolling_mean, boost::accumulators::tag::rolling_count>> CaptureIntervalStats_t;
  CaptureIntervalStats_t m_captureIntervalStats;

  void setCaptureDurationNs(uint64_t captureDurationNs);

  bool didAdjustCaptureTimingThisFrame() const { return m_didAdjustCaptureTimingThisFrame; }

  // Shared camera provider
  Argus::UniqueObj<Argus::CameraProvider> m_cameraProvider;

  struct BufferPool {
    struct Entry {
      Entry() = default;

      // Argus's per-buffer handle. The Argus::Buffer wraps an EGLImage that
      // is itself a window over the NvBufSurface owned by `rhiSurface`.
      // Argus writes into the NvBuf; VK + CUDA read it through `rhiSurface`.
      Argus::Buffer* argusBuffer = nullptr;
      EGLImageKHR eglImage = nullptr;

      // Owns the underlying NvBufSurface + VK + CUDA imports. Exposes the
      // sampled VkImageView (via inherited RHISurfaceVK) and per-plane CUDA
      // texture objects matching the IArgusCamera contract.
      ArgusCameraSurfaceVK::ptr rhiSurface;
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


  // Per-sensor objects
  struct SensorData {
    Argus::CameraDevice* m_cameraDevice = nullptr;
    Argus::OutputStream* m_outputStream = nullptr;

    BufferPool m_bufferPool;

    // Which buffers need to be released to the stream next readFrame
    Argus::Buffer* m_releaseBuffer = nullptr;

    uint32_t m_sessionIdx = 0;

    uint32_t m_captureFailureCount = 0;
    bool hasCaptureFailed() const { return m_captureFailureCount >= 3; }
  };


  std::vector<SensorData> m_perSensorData;

  // Shared sensor mode from m_perSensorData[0].m_cameraDevice
  Argus::SensorMode* m_sensorMode = nullptr;

  // Sessions and per-session objects
  struct SessionData {
    bool m_sessionCaptureFailed = false;
    Argus::CaptureSession* m_captureSession = nullptr;
    Argus::Request* m_captureRequest = nullptr;
    Argus::EventQueue* m_completionEventQueue = nullptr; // for EVENT_TYPE_CAPTURE_COMPLETE
    int64_t m_durationSkew_ns = 0;
  };
  std::vector<SessionData> m_perSessionData;


  uint32_t m_streamsPerSession = 1;
  virtual size_t sessionCount() const { return m_perSessionData.size(); }
  virtual size_t sessionIndexForStream(size_t streamIdx) const { return m_perSensorData[streamIdx].m_sessionIdx; }

  // Inter-session timing data
  struct SessionTimingData {
    float timestampDelta[4] = {0}; // in milliseconds; relative to the oldest sensor timestamp from any stream in any session.
  };
  ScrollingBuffer<SessionTimingData> m_sessionTimingData = ScrollingBuffer<SessionTimingData>(512);

  // Per-sensor timing data
  struct SensorTimingData {
    float frameAge[8] = {0}; // in milliseconds; age of frame at time of capture (diff between current time and sensor timestamp)
  };
  ScrollingBuffer<SensorTimingData> m_sensorTimingData = ScrollingBuffer<SensorTimingData>(512);


  bool m_usingExternalSync = false;
  bool m_adjustSessionSkew = true;

  bool m_adjustCaptureInterval = false;

  mutable RHISurface::ptr m_tmpBlitSurface;
  mutable RHIRenderTarget::ptr m_tmpBlitRT;

  // Shared ycbcr-aware sampler for the camera streams. Created once in the
  // ctor (NV12 / BT.709 / narrow range, matching Argus output for this
  // hardware). The same conversion is referenced by every ArgusCameraSurfaceVK's
  // VkImageView and is declared as the immutable sampler for every camera
  // pipeline's "imageTex" binding by main.cpp.
  RHISampler::ptr m_cameraSampler;

  // noncopyable
  ArgusCamera(const ArgusCamera&);
  ArgusCamera& operator=(const ArgusCamera&);
};
