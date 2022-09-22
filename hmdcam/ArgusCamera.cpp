#include "ArgusCamera.h"
#include "ArgusHelpers.h"
#include "common/EnvVar.h"
#include "common/Timing.h"
#include "imgui.h"
#include "rhi/gl/GLCommon.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/cuda/RHICVInterop.h"
#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <nvbuf_utils.h>
#include <cudaEGL.h>
#include <opencv2/cvconfig.h>
#ifdef HAVE_VPI2
#include "common/VPIUtil.h"
#include <vpi/Image.h>
#endif // HAVE_VPI2

#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/mean.hpp>

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)
//#define FRAME_WAIT_TIME_STATS 1

static const size_t kBufferCount = 8;

extern RHIRenderPipeline::ptr camTexturedQuadPipeline;
extern FxAtomicString ksNDCQuadUniformBlock;
extern FxAtomicString ksImageTex;

IArgusCamera::IArgusCamera() {}
IArgusCamera::~IArgusCamera() {}

ArgusCamera::ArgusCamera(EGLDisplay display_, EGLContext context_, double framerate) :
  m_display(display_), m_context(context_),
  m_shouldResubmitCaptureRequest(false),
  m_captureIsRepeating(false),
  m_minAcRegionWidth(0),
  m_minAcRegionHeight(0),
  m_captureIntervalStats(boost::accumulators::tag::rolling_window::window_size = m_adjustCaptureEvalWindowFrames) {

  m_targetCaptureIntervalNs = 1000000000.0 / framerate;


  m_cameraProvider.reset(Argus::CameraProvider::create());
  Argus::ICameraProvider* iCameraProvider = Argus::interface_cast<Argus::ICameraProvider>(m_cameraProvider.get());
  if (!iCameraProvider) {
    die("Failed to get ICameraProvider interface");
  }
  printf("Argus Version: %s\n", iCameraProvider->getVersion().c_str());

  iCameraProvider->getCameraDevices(&m_cameraDevices);
  if (m_cameraDevices.empty()) {
    die("No camera devices are available");
  }

  int maxSensors = 0;
  if (readEnvironmentVariable("ARGUS_MAX_SENSORS", maxSensors)) {
    if (m_cameraDevices.size() > maxSensors) {
      printf("DEBUG: Trimming sensor list from ARGUS_MAX_SENSORS=%d env\n", maxSensors);
      m_cameraDevices.resize(maxSensors);
    }
  }
  if (readEnvironmentVariable("ARGUS_STREAMS_PER_SESSION", m_streamsPerSession)) {
    printf("DEBUG: Using %u streams per session\n", m_streamsPerSession);
  }

  // Get the selected camera device and sensor mode.
  for (size_t cameraIdx = 0; cameraIdx < m_cameraDevices.size(); ++cameraIdx) {
    printf("Sensor %zu:\n", cameraIdx);
    ArgusHelpers::printCameraDeviceInfo(m_cameraDevices[cameraIdx], "  ");
  }

  // Pick a sensor mode from the first camera, which will be applied to all cameras
  Argus::ICameraProperties *iCameraProperties = Argus::interface_cast<Argus::ICameraProperties>(m_cameraDevices[0]);
  Argus::SensorMode* sensorMode = NULL;
  {
    // Select sensor mode. Pick the fastest mode (smallest FrameDurationRange.min) with the largest pixel area.
    uint64_t bestFrameDurationRangeMin = UINT64_MAX;
    uint64_t bestPixelArea = 0;

    std::vector<Argus::SensorMode*> sensorModes;
    iCameraProperties->getAllSensorModes(&sensorModes);
    for (size_t modeIdx = 0; modeIdx < sensorModes.size(); ++modeIdx) {
      Argus::SensorMode* sensorModeCandidate = sensorModes[modeIdx];
      Argus::ISensorMode *iSensorModeCandidate = Argus::interface_cast<Argus::ISensorMode>(sensorModeCandidate);

      uint64_t pixelArea = iSensorModeCandidate->getResolution().width() * iSensorModeCandidate->getResolution().height();

      if ((iSensorModeCandidate->getFrameDurationRange().min() < bestFrameDurationRangeMin) || // faster mode
        ((iSensorModeCandidate->getFrameDurationRange().min() == bestFrameDurationRangeMin) && (pixelArea > bestPixelArea))) /*same speed, more pixels*/ {
        bestFrameDurationRangeMin = iSensorModeCandidate->getFrameDurationRange().min();
        bestPixelArea = pixelArea;
        sensorMode = sensorModeCandidate;
      }
    }

    {
      char* e = getenv("ARGUS_MODE");
      if (e) {
        int modeIdx = atoi(e);
        if (modeIdx >= 0 && modeIdx < sensorModes.size()) {
          printf("Overriding mode selection to index %d by ARGUS_MODE environment variable\n", modeIdx);
          sensorMode = sensorModes[modeIdx];
        }
      }
    }

  }

  if (!sensorMode)
    die("Unable to select a sensor mode");

  Argus::ISensorMode *iSensorMode = Argus::interface_cast<Argus::ISensorMode>(sensorMode);
  assert(iSensorMode);

  printf("Selected sensor mode:\n");
  ArgusHelpers::printSensorModeInfo(sensorMode, "-- ");
  m_streamWidth = iSensorMode->getResolution().width();
  m_streamHeight = iSensorMode->getResolution().height();

  // Save minimum autocontrol region size
#if L4T_RELEASE_MAJOR < 34
  m_minAcRegionWidth = m_minAcRegionHeight = 64; // TODO just guessing here since this API didn't exist yet
#else
  Argus::Size2D<uint32_t> minAcRegionSize = iCameraProperties->getMinAeRegionSize();
  m_minAcRegionWidth = minAcRegionSize.width();
  m_minAcRegionHeight = minAcRegionSize.height();
#endif


  // Determine how many capture sessions we need
  size_t requiredCaptureSessions = (m_cameraDevices.size() + (m_streamsPerSession - 1)) / m_streamsPerSession;

  // Create capture sessions and per-session objects
  for (size_t sessionIdx = 0; sessionIdx < requiredCaptureSessions; ++sessionIdx) {
    std::vector<Argus::CameraDevice*> sessionDevices;

    // Populate device set for this session
    printf("Session %zu devices: ", sessionIdx);
    for (size_t cameraIdx = 0; cameraIdx < m_cameraDevices.size(); ++cameraIdx) {
      if (sessionIndexForCamera(cameraIdx) == sessionIdx) {
        sessionDevices.push_back(m_cameraDevices[cameraIdx]);
        printf("%zu ", cameraIdx);
      }
    }
    printf("\n");

    // Create the capture session
    Argus::CaptureSession* session = iCameraProvider->createCaptureSession(sessionDevices);
    m_captureSessions.push_back(session);

    Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(session);
    Argus::IEventProvider *iEventProvider = Argus::interface_cast<Argus::IEventProvider>(session);
    if (!iCaptureSession || !iEventProvider)
        die("Failed to create CaptureSession");

    Argus::EventQueue* completionEventQueue = iEventProvider->createEventQueue( {Argus::EVENT_TYPE_CAPTURE_COMPLETE });
    assert(completionEventQueue);
    m_sessionCompletionEventQueues.push_back(completionEventQueue);

    // Create capture request, set the sensor mode, and enable the output streams.
    Argus::Request* request = iCaptureSession->createRequest();
    m_sessionCaptureRequests.push_back(request);

    if (!request)
        die("Failed to create Request");
    Argus::ISourceSettings *iSourceSettings = Argus::interface_cast<Argus::ISourceSettings>(request);
    if (!iSourceSettings)
        die("Failed to get source settings request interface");
    iSourceSettings->setSensorMode(sensorMode);
  }


  // Create the per-camera OutputStreams and textures.
  m_bufferPools.resize(m_cameraDevices.size());
  m_releaseBuffers.resize(m_cameraDevices.size(), NULL);

  for (size_t cameraIdx = 0; cameraIdx < m_cameraDevices.size(); ++cameraIdx) {
    // Create the OutputStreamSettings object for a buffer OutputStream
    Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(m_captureSessions[sessionIndexForCamera(cameraIdx)]);

    Argus::UniqueObj<Argus::OutputStreamSettings> streamSettings(iCaptureSession->createOutputStreamSettings(Argus::STREAM_TYPE_BUFFER));
    Argus::IBufferOutputStreamSettings *iBufferOutputStreamSettings = Argus::interface_cast<Argus::IBufferOutputStreamSettings>(streamSettings);
    Argus::IOutputStreamSettings* iOutputStreamSettings = Argus::interface_cast<Argus::IOutputStreamSettings>(streamSettings);
    assert(iBufferOutputStreamSettings && iOutputStreamSettings);

    // Configure the OutputStream to use the EGLImage BufferType.
    iBufferOutputStreamSettings->setBufferType(Argus::BUFFER_TYPE_EGL_IMAGE);
    iBufferOutputStreamSettings->setMetadataEnable(true);

    iOutputStreamSettings->setCameraDevice(m_cameraDevices[cameraIdx]);
    Argus::OutputStream* outputStream = iCaptureSession->createOutputStream(streamSettings.get());
    m_outputStreams.push_back(outputStream);

    Argus::IBufferOutputStream *iBufferOutputStream = Argus::interface_cast<Argus::IBufferOutputStream>(outputStream);
    if (!iBufferOutputStream)
        die("Failed to create BufferOutputStream");

    // Create the BufferSettings object to configure Buffer creation.
    Argus::UniqueObj<Argus::BufferSettings> bufferSettings(iBufferOutputStream->createBufferSettings());
    Argus::IEGLImageBufferSettings *iBufferSettings = Argus::interface_cast<Argus::IEGLImageBufferSettings>(bufferSettings);
    iBufferSettings->setEGLDisplay(m_display);

    // Allocate native buffers, create the Argus::Buffer for each EGLImage, and release to stream for initial capture use.
    for (size_t i = 0; i < kBufferCount; i++) {
      BufferPool::Entry b;

      NvBufferCreateParams inputParams = {0};

      inputParams.width = m_streamWidth;
      inputParams.height = m_streamHeight;
      inputParams.layout = NvBufferLayout_Pitch;
      inputParams.colorFormat = NvBufferColorFormat_NV12_ER;
      inputParams.payloadType = NvBufferPayload_SurfArray;
      inputParams.nvbuf_tag = NvBufferTag_CAMERA;

      if (NvBufferCreateEx(&b.nativeBuffer, &inputParams)) {
        die("NvBufferCreateEx failed");
      }

      b.eglImage = NvEGLImageFromFd(m_display, b.nativeBuffer);
      b.rhiSurface = RHIEGLImageSurfaceGL::newTextureExternalOES(b.eglImage, m_streamWidth, m_streamHeight);

      iBufferSettings->setEGLImage(b.eglImage);
      b.argusBuffer = iBufferOutputStream->createBuffer(bufferSettings.get());
      if (!b.argusBuffer)
          die("Failed to create Buffer");

      if (iBufferOutputStream->releaseBuffer(b.argusBuffer) != Argus::STATUS_OK)
          die("Failed to release Buffer for capture use");

      CUDA_CHECK(cuGraphicsEGLRegisterImage(&b.cudaResource, b.eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY));
      CUDA_CHECK(cuGraphicsResourceGetMappedEglFrame(&b.eglFrame, b.cudaResource, 0, 0));

      int cvLumaFormat = CV_8U;
      if (i == 0) { // only report for the first buffer created
        // Typical eglColorFormat is CU_EGL_COLOR_FORMAT_YUV420_SEMIPLANAR_ER (0x26)
        // Surface [0]: Y, extended range.
        // Surface [1]: UV, with VU byte ordering, U/V width = 1/2 Y width, U/V height = 1/2 Y height.

        printf("Stream [%zu]: frameType=%s cuFormat=0x%x eglColorFormat=0x%x width=%u height=%u numChannels=%u planeCount=%u \n", cameraIdx,
          b.eglFrame.frameType == CU_EGL_FRAME_TYPE_ARRAY ? "Array" : "Pitch", b.eglFrame.cuFormat, b.eglFrame.eglColorFormat, b.eglFrame.width, b.eglFrame.height, b.eglFrame.numChannels, b.eglFrame.planeCount);
        switch (b.eglFrame.cuFormat) {
          case CU_AD_FORMAT_UNSIGNED_INT8: cvLumaFormat = CV_8U; break;
          case CU_AD_FORMAT_UNSIGNED_INT16: cvLumaFormat = CV_16U; break;
          case CU_AD_FORMAT_UNSIGNED_INT32: cvLumaFormat = CV_32S; break; // no real CV_32U type, so we cheat with CV_32S
          case CU_AD_FORMAT_SIGNED_INT8: cvLumaFormat = CV_8S; break;
          case CU_AD_FORMAT_SIGNED_INT16: cvLumaFormat = CV_16S; break;
          case CU_AD_FORMAT_SIGNED_INT32: cvLumaFormat = CV_32S; break;

          case CU_AD_FORMAT_HALF: cvLumaFormat = CV_16F; break;
          case CU_AD_FORMAT_FLOAT: cvLumaFormat = CV_32F; break;
          default:
            assert(false && "Unhandled cuFormat");
        }
      }

      // Create CUtexObject wrapper over the luma plane
      assert(b.eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH && b.eglFrame.frame.pPitch[0] != nullptr);
      {
        CUDA_RESOURCE_DESC resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = CU_RESOURCE_TYPE_PITCH2D;
        resDesc.res.pitch2D.devPtr       = (CUdeviceptr) b.eglFrame.frame.pPitch[0];
        resDesc.res.pitch2D.format       = b.eglFrame.cuFormat;
        resDesc.res.pitch2D.numChannels  = b.eglFrame.numChannels;
        resDesc.res.pitch2D.width        = b.eglFrame.width;
        resDesc.res.pitch2D.height       = b.eglFrame.height;
        resDesc.res.pitch2D.pitchInBytes = b.eglFrame.pitch;

        CUDA_TEXTURE_DESC texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
        // texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES; // optional
        texDesc.maxAnisotropy = 1;

        CUDA_CHECK(cuTexObjectCreate(&b.cudaLumaTexObject, &resDesc, &texDesc, /*resourceViewDescriptor=*/ nullptr));
      }

#ifdef HAVE_VPI2
      VPIImageData vid;
      memset(&vid, 0, sizeof(vid));
      vid.buffer.fd = b.nativeBuffer;
      vid.bufferType = VPI_IMAGE_BUFFER_NVBUFFER;
      VPI_CHECK(vpiImageCreateWrapper(&vid, NULL, /*flags=*/ 0, &b.vpiImage));

      if (i == 0) { // only report for the first buffer created
        VPIImageFormat imageFormat;
        VPI_CHECK(vpiImageGetFormat(b.vpiImage, &imageFormat) );
        printf("Stream [%zu]: VPIImageFormat is %s\n", cameraIdx, vpiImageFormatGetName(imageFormat));
      }
#endif

      m_bufferPools[cameraIdx].buffers.push_back(b);
    }

    // Enable the output stream on the associated session capture request
    Argus::IRequest *iRequest = Argus::interface_cast<Argus::IRequest>(m_sessionCaptureRequests[sessionIndexForCamera(cameraIdx)]);
    iRequest->enableOutputStream(outputStream);
  }

  // Set up all of the per-stream metadata containers
  m_frameMetadata.resize(m_cameraDevices.size());

  // Update autocontrol settings
  setExposureCompensation(m_exposureCompensation);
  setAcRegion(m_acRegionCenter, m_acRegionSize);

  m_captureDurationMinNs = iSensorMode->getFrameDurationRange().min();
  m_captureDurationMaxNs = iSensorMode->getFrameDurationRange().max();

  // Set the initial capture duration to the requested frame interval. This will be wrong since there's some overhead;
  // we'll recompute it later once we start getting back capture timestamps.
  setCaptureDurationNs(m_targetCaptureIntervalNs);
}

void ArgusCamera::setRepeatCapture(bool value) {
  if (m_captureIsRepeating == value)
    return;

  m_captureIsRepeating = value;

  if (m_captureIsRepeating) {
    // Start a repeating capture
    for (size_t sessionIdx = 0; sessionIdx < m_captureSessions.size(); ++sessionIdx) {
      Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(m_captureSessions[sessionIdx]);
      if (iCaptureSession->repeat(m_sessionCaptureRequests[sessionIdx]) != Argus::STATUS_OK)
        die("Failed to start repeat capture request");
    }
  } else {
    // Issue all stop-repeat requests and wait for the sessions to become idle
    for (size_t sessionIdx = 0; sessionIdx < m_captureSessions.size(); ++sessionIdx) {
      Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(m_captureSessions[sessionIdx]);
      iCaptureSession->cancelRequests();
    }
    // Give the sessions time to return to idle
    for (size_t sessionIdx = 0; sessionIdx < m_captureSessions.size(); ++sessionIdx) {
      Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(m_captureSessions[sessionIdx]);
      iCaptureSession->waitForIdle(m_targetCaptureIntervalNs * 2);
    }
  }

}

ArgusCamera::~ArgusCamera() {
  for (Argus::OutputStream* outputStream : m_outputStreams)
    outputStream->destroy();
  m_outputStreams.clear();

  for (BufferPool& bp : m_bufferPools) {
    for (BufferPool::Entry& b : bp.buffers) {
      b.argusBuffer->destroy();
      cuTexObjectDestroy(b.cudaLumaTexObject);
      cuGraphicsUnregisterResource(b.cudaResource);
      eglDestroyImageKHR(m_display, b.eglImage);
      NvBufferDestroy(b.nativeBuffer);

#ifdef HAVE_VPI2
      vpiImageDestroy(b.vpiImage);
#endif
    }
  }
  m_bufferPools.clear();
}

bool ArgusCamera::readFrame() {
  static uint32_t s_frameCounter = 0;
  ++s_frameCounter;

#ifdef FRAME_WAIT_TIME_STATS
  static boost::accumulators::accumulator_set<double, boost::accumulators::stats<
        boost::accumulators::tag::min,
        boost::accumulators::tag::max,
        boost::accumulators::tag::mean,
        boost::accumulators::tag::median
    > > s_frameWaitTimeStats;
#endif

  if (!m_captureIsRepeating) {
    for (size_t sessionIdx = 0; sessionIdx < m_captureSessions.size(); ++sessionIdx) {
      Argus::Status status;
      Argus::interface_cast<Argus::ICaptureSession>(m_captureSessions[sessionIdx])->capture(m_sessionCaptureRequests[sessionIdx], Argus::TIMEOUT_INFINITE, &status);
      if (status != Argus::STATUS_OK)
        die("ArgusCamera::readFrame(): Failed to request capture");
    }
  }


#ifdef FRAME_WAIT_TIME_STATS
  uint64_t eventWaitStart = currentTimeNs();
#endif

  // Service CaptureSession event queue and wait for capture completed event here
  // that should be able to smooth out some of the jitter without missing frames
  for (size_t sessionIdx = 0; sessionIdx < m_captureSessions.size(); ++sessionIdx) {
    Argus::interface_cast<Argus::IEventProvider>(m_captureSessions[sessionIdx])->waitForEvents(m_sessionCompletionEventQueues[sessionIdx], m_targetCaptureIntervalNs / 2);
  }

#ifdef FRAME_WAIT_TIME_STATS
  uint64_t eventWaitEnd = currentTimeNs();
  if (m_captureIsRepeating) {
    s_frameWaitTimeStats(static_cast<double>(eventWaitEnd - eventWaitStart) / 1000000.0);

    if ((s_frameCounter & 0x7f) == 0x7f) {
      printf("Frame wait-time: min=%.3g max=%.3g mean=%.3g median=%.3g\n",
        boost::accumulators::min(s_frameWaitTimeStats),
        boost::accumulators::max(s_frameWaitTimeStats),
        boost::accumulators::mean(s_frameWaitTimeStats),
        boost::accumulators::median(s_frameWaitTimeStats));
        s_frameWaitTimeStats = {};
    }
  }
#endif

  static uint64_t previousCaptureCompletionTimestamp = 0;
  if (m_captureIsRepeating) {
    // Pump event queues for all sessions
    for (size_t sessionIdx = 0; sessionIdx < m_captureSessions.size(); ++sessionIdx) {
      while (true) {
        const Argus::Event* ev = Argus::interface_cast<Argus::IEventQueue>(m_sessionCompletionEventQueues[sessionIdx])->getNextEvent();

        if (!ev)
          break;

        const Argus::IEvent* iev = Argus::interface_cast<const Argus::IEvent>(ev);
        if (iev->getEventType() == Argus::EVENT_TYPE_CAPTURE_COMPLETE) {
          if (previousCaptureCompletionTimestamp) {
            int64_t ts_delta_us = iev->getTime() - previousCaptureCompletionTimestamp;
            m_captureIntervalStats(static_cast<double>(ts_delta_us * 1000 /*convert to ns*/));
          }
          previousCaptureCompletionTimestamp = iev->getTime();
        }
        //if (iev->getEventType() == Argus::EVENT_TYPE_CAPTURE_STARTED)

      }
    }
  } else {
    // Don't track timestamp deltas for single shot captures
    previousCaptureCompletionTimestamp = 0;
  }

  bool res = true;
  for (size_t cameraIdx = 0; cameraIdx < m_cameraDevices.size(); ++cameraIdx) {
    Argus::IBufferOutputStream *iBufferOutputStream = Argus::interface_cast<Argus::IBufferOutputStream>(m_outputStreams[cameraIdx]);
    Argus::Status status = Argus::STATUS_OK;
    Argus::Buffer* buffer = iBufferOutputStream->acquireBuffer(Argus::TIMEOUT_INFINITE, &status);
    m_bufferPools[cameraIdx].setActiveBufferIndex(buffer);

    // Clean up previous capture's buffer and track this one to be released next round
    if (m_releaseBuffers[cameraIdx])
      iBufferOutputStream->releaseBuffer(m_releaseBuffers[cameraIdx]);

    m_releaseBuffers[cameraIdx] = buffer;

    Argus::IEGLImageBuffer* eglImageBuffer = Argus::interface_cast<Argus::IEGLImageBuffer>(buffer);
    assert(eglImageBuffer);

    Argus::IBuffer* iBuffer = Argus::interface_cast<Argus::IBuffer>(buffer);
    assert(iBuffer);

    const Argus::ICaptureMetadata* iMetadata = Argus::interface_cast<const Argus::ICaptureMetadata>(iBuffer->getMetadata());

    if (!iMetadata) {
      printf("ArgusCamera::readFrame(): Failed to read metadata for camera index %zu\n", cameraIdx);
      continue;
    }

    // Update metadata fields for this frame
    m_frameMetadata[cameraIdx].sensorTimestamp = iMetadata->getSensorTimestamp();
    m_frameMetadata[cameraIdx].frameDurationNs = iMetadata->getFrameDuration();
    m_frameMetadata[cameraIdx].sensorExposureTimeNs = iMetadata->getSensorExposureTime();
    m_frameMetadata[cameraIdx].sensorSensitivityISO = iMetadata->getSensorSensitivity();
    m_frameMetadata[cameraIdx].ispDigitalGain = iMetadata->getIspDigitalGain();
    m_frameMetadata[cameraIdx].sensorAnalogGain = iMetadata->getSensorAnalogGain();
  }

  m_didAdjustCaptureIntervalThisFrame = false;
  if (m_adjustCaptureInterval && m_captureIsRepeating && (((++m_samplesAtCurrentDuration) > m_adjustCaptureCooldownFrames) && m_previousSensorTimestampNs)) {

    if (boost::accumulators::rolling_count(m_captureIntervalStats) >= m_adjustCaptureEvalWindowFrames) {

      int64_t durationToTSDeltaOffset = boost::accumulators::rolling_mean(m_captureIntervalStats) - m_currentCaptureDurationNs;

      // printf("Capture duration % .6f -> interval % .6f (duration-to-interval offset: %ld ns)\n", static_cast<double>(m_currentCaptureDurationNs) / 1000000.0, static_cast<double>(boost::accumulators::rolling_mean(m_captureIntervalStats)) / 1000000.0, durationToTSDeltaOffset);

      int64_t targetDuration = m_targetCaptureIntervalNs - durationToTSDeltaOffset;
      // Clamp the offset to a reasonable small value so that a timestamp discontinuity doesn't cause a massive overshoot
      const int64_t offsetMax = 500000; // 500 microseconds
      int64_t targetOffset = std::min<int64_t>(std::max<int64_t>(targetDuration - m_currentCaptureDurationNs, -offsetMax), offsetMax);
      // printf("durationToTSDeltaOffset %ld targetDuration %ld targetOffset %ld\n", durationToTSDeltaOffset, targetDuration, targetOffset);

      // Perform an adjustment if we're off by at least 50 microseconds
      if (std::abs(targetOffset) > 50000) {
        // Clamp new duration to sensor mode limits
        int64_t newDuration = std::min<int64_t>(std::max<int64_t>(m_currentCaptureDurationNs + (targetOffset / 64), m_captureDurationMinNs), m_captureDurationMaxNs);
        // printf("Capture duration adjust %ld (%ld -> %ld)\n", targetOffset/64, m_currentCaptureDurationNs, newDuration);
        setCaptureDurationNs(newDuration);
      }
    }
  }

  if (m_shouldResubmitCaptureRequest && m_captureIsRepeating) {
    // Resubmit repeating capture request for dirty controls/settings
    for (size_t sessionIdx = 0; sessionIdx < m_captureSessions.size(); ++sessionIdx) {
      if (Argus::interface_cast<Argus::ICaptureSession>(m_captureSessions[sessionIdx])->repeat(m_sessionCaptureRequests[sessionIdx]) != Argus::STATUS_OK)
        die("Failed to update repeat capture request");
    }
  }

  m_shouldResubmitCaptureRequest = false;
  m_previousSensorTimestampNs = m_frameMetadata[0].sensorTimestamp;

  return res;
}

void ArgusCamera::setCaptureDurationNs(uint64_t captureDurationNs) {
  for (size_t sessionIdx = 0; sessionIdx < m_captureSessions.size(); ++sessionIdx) {
    Argus::interface_cast<Argus::ISourceSettings>(m_sessionCaptureRequests[sessionIdx])->setFrameDurationRange(captureDurationNs);
  }
  m_currentCaptureDurationNs = captureDurationNs;
  m_samplesAtCurrentDuration = 0;
  m_didAdjustCaptureIntervalThisFrame = true;
  m_shouldResubmitCaptureRequest = true;
}

int64_t ArgusCamera::captureDurationOffset() const {
  return static_cast<int64_t>(m_currentCaptureDurationNs) - static_cast<int64_t>(m_targetCaptureIntervalNs);
}

void ArgusCamera::setCaptureDurationOffset(int64_t ns) {
  if (captureDurationOffset() == ns)
    return; // no change

  setCaptureDurationNs(static_cast<uint64_t>(static_cast<int64_t>(m_targetCaptureIntervalNs) + ns));
}

void ArgusCamera::stop() {
  setRepeatCapture(false);
}


void ArgusCamera::setExposureCompensation(float stops) {
  for (size_t sessionIdx = 0; sessionIdx < m_captureSessions.size(); ++sessionIdx) {
    Argus::IAutoControlSettings* iAutoControlSettings = Argus::interface_cast<Argus::IAutoControlSettings>(Argus::interface_cast<Argus::IRequest>(m_sessionCaptureRequests[sessionIdx])->getAutoControlSettings());
    assert(iAutoControlSettings);

    iAutoControlSettings->setExposureCompensation(stops);
  }
  m_exposureCompensation = stops;
  m_shouldResubmitCaptureRequest = true;
}

void ArgusCamera::setAcRegion(const glm::vec2& center, const glm::vec2& size) {
  m_acRegionCenter = glm::clamp(center, glm::vec2(0.0f), glm::vec2(1.0f));
  m_acRegionSize = glm::clamp(size, glm::vec2(0.0f), glm::vec2(1.0f));

  const glm::vec2 streamSize = glm::vec2(m_streamWidth, m_streamHeight);

  glm::vec2 acRegionSizePx = glm::max(glm::ceil(m_acRegionSize * streamSize), glm::vec2(m_minAcRegionWidth, m_minAcRegionHeight));
  glm::vec2 acRegionCenterPx = glm::ceil(m_acRegionCenter * streamSize);

  glm::vec2 lt = glm::max(glm::floor(acRegionCenterPx - (acRegionSizePx * 0.5f)), glm::vec2(0.0f, 0.0f));
  glm::vec2 rb = glm::min(glm::ceil(acRegionCenterPx + (acRegionSizePx * 0.5f)), streamSize);

  Argus::AcRegion region;
  region.weight() = 1.0f;

  // Round to 8px block size
  region.left() = ((uint32_t) lt[0]) & (~(0x07));
  region.top() = ((uint32_t) lt[1]) & (~(0x07));
  region.right() = std::min<uint32_t>((((uint32_t) rb[0]) + 7) & (~(0x07)), m_streamWidth);
  region.bottom() = std::min<uint32_t>((((uint32_t) rb[1]) + 7) & (~(0x07)), m_streamHeight);

  std::vector<Argus::AcRegion> regions;
  regions.push_back(region);

  for (size_t sessionIdx = 0; sessionIdx < m_captureSessions.size(); ++sessionIdx) {
    Argus::IAutoControlSettings* iAutoControlSettings = Argus::interface_cast<Argus::IAutoControlSettings>(Argus::interface_cast<Argus::IRequest>(m_sessionCaptureRequests[sessionIdx])->getAutoControlSettings());
    assert(iAutoControlSettings);

    Argus::Status status = iAutoControlSettings->setAeRegions(regions);
    if (status == Argus::STATUS_OK)
      status = iAutoControlSettings->setAwbRegions(regions);
    if (status != Argus::STATUS_OK) {
      printf("Unable to set AC region to LTRB=%u, %u, %u, %u: status code %d\n", region.left(), region.top(), region.right(), region.bottom(), status);
    }
    // TODO might also want to do AF regions here if we ever get an autofocus camera
  }


  m_shouldResubmitCaptureRequest = true;
}

bool ArgusCamera::renderPerformanceTuningIMGUI() {
  bool settingsDirty = false;

  settingsDirty |= ImGui::Checkbox("Auto-adjust capture interval", &m_adjustCaptureInterval);

  if (m_adjustCaptureInterval) {
    ImGui::SliderInt("Adjustment Cooldown Frames", &m_adjustCaptureCooldownFrames, 1, 64);
    if (ImGui::SliderInt("Adjustment Tgt Eval Window", &m_adjustCaptureEvalWindowFrames, 1, 256)) {
      m_captureIntervalStats = CaptureIntervalStats_t(boost::accumulators::tag::rolling_window::window_size = m_adjustCaptureEvalWindowFrames);
    }
  }

  float offsetUs = static_cast<float>(captureDurationOffset()) / 1000.0f;
  if (ImGui::SliderFloat("Offset (us)", &offsetUs, -1000.0f, 1000.0f, "%.1f", ImGuiSliderFlags_None)) {
    setCaptureDurationOffset(static_cast<int64_t>(offsetUs * 1000.0f));
    settingsDirty = true;
  }


  return settingsDirty;
}

cv::cuda::GpuMat ArgusCamera::gpuMatGreyscale(size_t sensorIdx) {
  const CUeglFrame& eglFrame = m_bufferPools[sensorIdx].activeBuffer().eglFrame;
  return cv::cuda::GpuMat(eglFrame.height, eglFrame.width, CV_8U, eglFrame.frame.pPitch[0], eglFrame.pitch);
}

/*
void ArgusCamera::populateGpuMat(size_t sensorIdx, cv::cuda::GpuMat& gpuMat, const cv::cuda::Stream& stream) {
  CUgraphicsResource pReadResource = NULL;
  CUresult status = cuGraphicsEGLRegisterImage(&pReadResource, m_currentEglImages[sensorIdx], CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY);
  if (status != CUDA_SUCCESS)
    die("cuGraphicsEGLRegisterImage failed: %d\n", status);

  CUeglFrame eglFrame;
  status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pReadResource, 0, 0);
  if (status != CUDA_SUCCESS) {
    die("cuGraphicsSubResourceGetMappedArray failed: %d\n", status);
  }

  // TODO optionally support NV12 -> RGB format conversion

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));

  assert(eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH);
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.srcDevice = (CUdeviceptr) eglFrame.frame.pPitch[0];
  copyDescriptor.srcPitch = eglFrame.pitch;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.dstDevice = (CUdeviceptr) gpuMat.cudaPtr();
  copyDescriptor.dstPitch = gpuMat.step;

  copyDescriptor.WidthInBytes = eglFrame.frame.width * gpuMat.elemSize();
  copyDescriptor.Height = copyHeight;

  CUStream streamPtr = (CUStream) stream.cudaPtr();
  if (streamPtr) {
    CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, streamPtr));
  } else {
    CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
  }

  cuGraphicsUnregisterResource(pReadResource);
}
*/

void ArgusCamera::populateGpuMat(size_t sensorIdx, cv::cuda::GpuMat& gpuMat, const cv::cuda::Stream& stream) {
#ifdef HAVE_CUDA // from opencv2/cvconfig.h
  if (!m_tmpBlitSurface) {
    m_tmpBlitSurface = rhi()->newTexture2D(streamWidth(), streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    m_tmpBlitRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor( { m_tmpBlitSurface } ));
  }

  glm::mat4 ub = glm::mat4(1.0f);
  ub[1][1] = -1.0f; // Y-flip for coordsys matching

  rhi()->beginRenderPass(m_tmpBlitRT, kLoadInvalidate);
  rhi()->bindRenderPipeline(camTexturedQuadPipeline);
  rhi()->loadTexture(ksImageTex, rgbTexture(sensorIdx), linearClampSampler);
  rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(glm::mat4));
  rhi()->drawNDCQuad();
  rhi()->endRenderPass(m_tmpBlitRT);

  RHICUDA::copySurfaceToGpuMat(m_tmpBlitSurface, gpuMat, const_cast<cv::cuda::Stream&>(stream));
#endif
}


// === Mock implementations ===

ArgusCameraMock::ArgusCameraMock(size_t sensorCount, unsigned int w, unsigned int h, double framerate) {
  m_streamWidth = w;
  m_streamHeight = h;
  m_targetCaptureIntervalNs = 1000000000.0 / framerate;

  m_frameMetadata.resize(sensorCount);
  for (size_t i = 0; i < m_frameMetadata.size(); ++i) {
    auto& md = m_frameMetadata[i];
    md.sensorTimestamp = 0;
    md.frameDurationNs = m_targetCaptureIntervalNs;
    md.sensorExposureTimeNs = m_targetCaptureIntervalNs;
    md.sensorSensitivityISO = 100;
    md.ispDigitalGain = 1.0f;
    md.sensorAnalogGain = 1.0f;
  }

  m_textures.resize(sensorCount);

  m_vpiImages.resize(sensorCount);
}

ArgusCameraMock::~ArgusCameraMock() {

}

bool ArgusCameraMock::readFrame() {
  uint64_t now = currentTimeNs();

  // Frame pacing
  uint64_t delta = now - m_previousFrameReadTime;
  if (delta < m_targetCaptureIntervalNs) {
    delayNs(m_targetCaptureIntervalNs - delta);
  }

  for (size_t i = 0; i < m_frameMetadata.size(); ++i) {
    m_frameMetadata[i].sensorTimestamp = now - m_targetCaptureIntervalNs;
  }
  for (size_t i = 0; i < m_textures.size(); ++i) {
    if (m_textures[i])
      continue;

    RHISurface::ptr srf = rhi()->newTexture2D(streamWidth(), streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    RHIRenderTarget::ptr rt = rhi()->compileRenderTarget(RHIRenderTargetDescriptor( { srf } ));

    switch (i) {
      case 0:
        rhi()->setClearColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)); break;
      case 1:
        rhi()->setClearColor(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)); break;
      case 2:
        rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)); break;
      case 3:
      default:
        rhi()->setClearColor(glm::vec4(1.0f, 0.0f, 1.0f, 1.0f)); break;
    };

    rhi()->beginRenderPass(rt, kLoadClear);
    rhi()->endRenderPass(rt);

    m_textures[i] = srf;

#ifdef HAVE_VPI2
    // Argus NVBuffers wrapped into VPIImage are VPI_IMAGE_FORMAT_NV12_ER
    VPI_CHECK(vpiImageCreate(streamWidth(), streamHeight(), VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_CUDA, &m_vpiImages[i]));
#endif
  }

  m_previousFrameReadTime = now;
  return true;
}

VPIImage ArgusCameraMock::vpiImage(size_t sensorIndex) const {
  return m_vpiImages[sensorIndex];
  return NULL;
}

cv::cuda::GpuMat ArgusCameraMock::gpuMatGreyscale(size_t sensorIdx) {
  assert(false && "Not implemented");
  return cv::cuda::GpuMat();
}

