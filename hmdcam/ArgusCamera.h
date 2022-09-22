#pragma once
#include <vector>
#include <epoxy/egl.h>
#include <Argus/Argus.h>
#include "rhi/egl/RHIEGLImageSurfaceGL.h"
#include "rhi/RHIRenderTarget.h"
#include "common/ICameraProvider.h"
#include <cuda.h>
#include <cudaEGL.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_count.hpp>


class IArgusCamera : public ICameraProvider {
public:
  IArgusCamera();
  virtual ~IArgusCamera();

  // === ICameraProvider (partial impl) ===
  virtual unsigned int streamWidth() const { return m_streamWidth; }
  virtual unsigned int streamHeight() const { return m_streamHeight; }

  // === IArgusCamera ===
  virtual CUgraphicsResource cudaGraphicsResource(size_t sensorIndex) const = 0;
  virtual bool readFrame() = 0;
  virtual void stop() = 0;
  virtual void setRepeatCapture(bool) = 0;
  virtual void setExposureCompensation(float stops) = 0;
  virtual void setAcRegion(const glm::vec2& center, const glm::vec2& size) = 0;

  virtual void setCaptureDurationOffset(int64_t ns) = 0;
  virtual int64_t captureDurationOffset() const = 0;

  virtual bool renderPerformanceTuningIMGUI() = 0;
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

class ArgusCamera : public IArgusCamera {
public:
  ArgusCamera(EGLDisplay, EGLContext, double framerate);
  virtual ~ArgusCamera();

  // === ICameraProvider ===
  virtual size_t streamCount() const { return m_bufferPools.size(); }
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const { return m_bufferPools[sensorIndex].activeBuffer().rhiSurface; }
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
  virtual void setAcRegion(const glm::vec2& center, const glm::vec2& size);

  virtual bool renderPerformanceTuningIMGUI();

  void setCaptureDurationOffset(int64_t ns);
  int64_t captureDurationOffset() const;

  // ====================

  void populateGpuMat(size_t sensorIndex, cv::cuda::GpuMat&, const cv::cuda::Stream&);

private:
  EGLDisplay m_display;
  EGLContext m_context;

  bool m_shouldResubmitCaptureRequest : 1;
  bool m_captureIsRepeating : 1;

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
  std::vector<Argus::CaptureSession*> m_captureSessions;
  std::vector<Argus::Request*> m_sessionCaptureRequests;
  std::vector<Argus::EventQueue*> m_sessionCompletionEventQueues; // for EVENT_TYPE_CAPTURE_COMPLETE

  static const size_t kCamerasPerSession = 2;
  static size_t sessionIndexForCamera(size_t cameraIdx) { return cameraIdx / kCamerasPerSession; }

  mutable RHISurface::ptr m_tmpBlitSurface;
  mutable RHIRenderTarget::ptr m_tmpBlitRT;

  // noncopyable
  ArgusCamera(const ArgusCamera&);
  ArgusCamera& operator=(const ArgusCamera&);
};


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
  virtual CUgraphicsResource cudaGraphicsResource(size_t sensorIndex) const { return nullptr; }
  virtual bool readFrame();

  virtual void stop() {}
  virtual void setRepeatCapture(bool) {}
  virtual void setExposureCompensation(float stops) {}
  virtual void setAcRegion(const glm::vec2& center, const glm::vec2& size) {}
  virtual bool renderPerformanceTuningIMGUI() { return false; }

  virtual void setCaptureDurationOffset(int64_t ns) {}
  virtual int64_t captureDurationOffset() const { return 0; }

protected:
  uint64_t m_previousFrameReadTime = 0;
  std::vector<RHISurface::ptr> m_textures;
  std::vector<VPIImage> m_vpiImages;
};

