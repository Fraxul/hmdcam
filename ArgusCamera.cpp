#include "ArgusCamera.h"
#include "ArgusHelpers.h"
#include "rhi/gl/GLCommon.h"
#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>

#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/mean.hpp>

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)
//#define FRAME_WAIT_TIME_STATS 1

static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

ArgusCamera::ArgusCamera(EGLDisplay display_, EGLContext context_, std::vector<unsigned int> cameraIds, double framerate) :
  m_display(display_), m_context(context_),
  m_cameraIds(cameraIds),
  m_captureIntervalStats(boost::accumulators::tag::rolling_window::window_size = 128),
  m_captureSession(NULL), m_captureRequest(NULL) {

  m_targetCaptureIntervalNs = 1000000000.0 / framerate;

  if (cameraIds.empty()) {
    die("No camera IDs provided");
  }

  m_cameraProvider.reset(Argus::CameraProvider::create());
  Argus::ICameraProvider* iCameraProvider = Argus::interface_cast<Argus::ICameraProvider>(m_cameraProvider.get());
  if (!iCameraProvider) {
    die("Failed to get ICameraProvider interface");
  }
  printf("Argus Version: %s\n", iCameraProvider->getVersion().c_str());

  // DEBUG: Dump the list of available cameras
  {
    std::vector<Argus::CameraDevice*> devices;
    iCameraProvider->getCameraDevices(&devices);
    printf("Devices (%zu):\n", devices.size());
    for (size_t deviceIdx = 0; deviceIdx < devices.size(); ++deviceIdx) {
      printf("[%zu] %p\n", deviceIdx, devices[deviceIdx]);
    }
  }

  // Get the selected camera device and sensor mode.
  for (size_t cameraIdx = 0; cameraIdx < m_cameraIds.size(); ++cameraIdx) {
    Argus::CameraDevice* cameraDevice = ArgusHelpers::getCameraDevice(m_cameraProvider.get(), cameraIds[cameraIdx]);
    if (!cameraDevice)
        die("Selected camera (index %zu, id %u) is not available", cameraIdx, cameraIds[cameraIdx]);
    m_cameraDevices.push_back(cameraDevice);

    printf("[%zu] Sensor %u:\n", cameraIdx, cameraIds[cameraIdx]);
    ArgusHelpers::printCameraDeviceInfo(cameraDevice, "  ");
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
  }

  if (!sensorMode)
    die("Unable to select a sensor mode");

  Argus::ISensorMode *iSensorMode = Argus::interface_cast<Argus::ISensorMode>(sensorMode);
  assert(iSensorMode);

  printf("Selected sensor mode:\n");
  ArgusHelpers::printSensorModeInfo(sensorMode, "-- ");
  m_streamWidth = iSensorMode->getResolution().width();
  m_streamHeight = iSensorMode->getResolution().height();


  // Create the capture session using the specified devices
  m_captureSession = iCameraProvider->createCaptureSession(m_cameraDevices);
  Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(m_captureSession);
  Argus::IEventProvider *iEventProvider = Argus::interface_cast<Argus::IEventProvider>(m_captureSession);
  if (!iCaptureSession || !iEventProvider)
      die("Failed to create CaptureSession");

  m_completionEventQueue = iEventProvider->createEventQueue( {Argus::EVENT_TYPE_CAPTURE_COMPLETE });
  assert(m_completionEventQueue);

  // Create the OutputStreamSettings object for an EGL OutputStream
  Argus::UniqueObj<Argus::OutputStreamSettings> streamSettings(iCaptureSession->createOutputStreamSettings(Argus::STREAM_TYPE_EGL));
  Argus::IEGLOutputStreamSettings *iEGLOutputStreamSettings = Argus::interface_cast<Argus::IEGLOutputStreamSettings>(streamSettings);
  Argus::IOutputStreamSettings* iOutputStreamSettings = Argus::interface_cast<Argus::IOutputStreamSettings>(streamSettings);
  assert(iEGLOutputStreamSettings && iOutputStreamSettings);

  // Configure the OutputStream to use the EGLImage BufferType.
  iEGLOutputStreamSettings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888);
  iEGLOutputStreamSettings->setResolution(Argus::Size2D<uint32_t>(m_streamWidth, m_streamHeight));
  iEGLOutputStreamSettings->setEGLDisplay(m_display);
  iEGLOutputStreamSettings->setMode(Argus::EGL_STREAM_MODE_MAILBOX);
  iEGLOutputStreamSettings->setMetadataEnable(true);

  // Create the per-camera OutputStreams and textures.
  for (size_t cameraIdx = 0; cameraIdx < m_cameraIds.size(); ++cameraIdx) {
    iOutputStreamSettings->setCameraDevice(m_cameraDevices[cameraIdx]);
    Argus::OutputStream* outputStream = iCaptureSession->createOutputStream(streamSettings.get());
    m_outputStreams.push_back(outputStream);

    Argus::IEGLOutputStream *iEGLOutputStream = Argus::interface_cast<Argus::IEGLOutputStream>(outputStream);
    if (!iEGLOutputStream)
        die("Failed to create EGLOutputStream");

    m_eglStreams.push_back(iEGLOutputStream->getEGLStream());

    // GL textures which will be associated with EGL images in readFrame()
    m_textures.push_back(new RHIEGLStreamSurfaceGL(m_streamWidth, m_streamHeight, kSurfaceFormat_RGBA8));
    glBindTexture(GL_TEXTURE_EXTERNAL_OES, m_textures[cameraIdx]->glId());

    // Connect the stream consumer. This must be done before starting the capture session, or libargus will return an invalid state error.
    if (!eglStreamConsumerGLTextureExternalKHR(m_display, m_eglStreams[cameraIdx]))
      die("Unable to connect GL as EGLStream consumer");

    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    // Set the acquire timeout to infinite.
    eglStreamAttribKHR(m_display, m_eglStreams[cameraIdx], EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, -1);
  }

  // Set up all of the per-stream metadata containers
  m_frameMetadata.resize(m_eglStreams.size());

  // Create capture request, set the sensor mode, and enable the output stream.s
  m_captureRequest = iCaptureSession->createRequest();
  Argus::IRequest *iRequest = Argus::interface_cast<Argus::IRequest>(m_captureRequest);
  if (!iRequest)
      die("Failed to create Request");
  for (Argus::OutputStream* outputStream : m_outputStreams) {
    iRequest->enableOutputStream(outputStream);
  }
  Argus::ISourceSettings *iSourceSettings = Argus::interface_cast<Argus::ISourceSettings>(m_captureRequest);
  if (!iSourceSettings)
      die("Failed to get source settings request interface");
  iSourceSettings->setSensorMode(sensorMode);

  m_captureDurationMinNs = iSensorMode->getFrameDurationRange().min();
  m_captureDurationMaxNs = iSensorMode->getFrameDurationRange().max();

  // Set the initial capture duration to the requested frame interval. This will be wrong since there's some overhead;
  // we'll recompute it later once we start getting back capture timestamps.
  setCaptureDurationNs(m_targetCaptureIntervalNs);

  m_captureIsRepeating = false;
}

void ArgusCamera::setRepeatCapture(bool value) {
  if (m_captureIsRepeating == value)
    return;

  m_captureIsRepeating = value;

  Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(m_captureSession);
  if (m_captureIsRepeating) {
    // Start a repeating capture
    if (iCaptureSession->repeat(m_captureRequest) != Argus::STATUS_OK)
      die("Failed to start repeat capture request");
  } else {
    iCaptureSession->stopRepeat();
    iCaptureSession->waitForIdle();
  }
}

ArgusCamera::~ArgusCamera() {
  for (Argus::OutputStream* outputStream : m_outputStreams)
    outputStream->destroy();
  m_outputStreams.clear();

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
    Argus::Status status;
    Argus::interface_cast<Argus::ICaptureSession>(m_captureSession)->capture(m_captureRequest, Argus::TIMEOUT_INFINITE, &status);
    if (status != Argus::STATUS_OK)
      die("ArgusCamera::readFrame(): Failed to request capture");
  }


#ifdef FRAME_WAIT_TIME_STATS
  uint64_t eventWaitStart = currentTimeNs();
#endif

  // Service CaptureSession event queue and wait for capture completed event here
  // that should be able to smooth out some of the jitter without missing frames
  Argus::interface_cast<Argus::IEventProvider>(m_captureSession)->waitForEvents(m_completionEventQueue, m_targetCaptureIntervalNs / 2);

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
    while (true) {
      const Argus::Event* ev = Argus::interface_cast<Argus::IEventQueue>(m_completionEventQueue)->getNextEvent();
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
  } else {
    // Don't track timestamp deltas for single shot captures
    previousCaptureCompletionTimestamp = 0;
  }


  bool res = true;
  for (size_t streamIdx = 0; streamIdx < m_eglStreams.size(); ++streamIdx) {
    const EGLStreamKHR& eglStream = m_eglStreams[streamIdx];
    res |= eglStreamConsumerAcquireKHR(m_display, eglStream);

    const Argus::ICaptureMetadata* iMetadata = NULL;

    // Try and acquire the metadata interface for this frame from the EGLStream
    Argus::UniqueObj<EGLStream::MetadataContainer> metadataContainer(EGLStream::MetadataContainer::create(m_display, eglStream));
    EGLStream::IArgusCaptureMetadata *iArgusCaptureMetadata = Argus::interface_cast<EGLStream::IArgusCaptureMetadata>(metadataContainer);

    if (iArgusCaptureMetadata) {
      iMetadata = Argus::interface_cast<const Argus::ICaptureMetadata>(iArgusCaptureMetadata->getMetadata());
    }

    if (!iMetadata) {
      printf("ArgusCamera::readFrame(): Failed to read metadata for stream index %zu\n", streamIdx);
      continue;
    }

    // Update metadata fields for this frame
    m_frameMetadata[streamIdx].sensorTimestamp = iMetadata->getSensorTimestamp();
    m_frameMetadata[streamIdx].sensorExposureTimeNs = iMetadata->getSensorExposureTime();
    m_frameMetadata[streamIdx].sensorSensitivityISO = iMetadata->getSensorSensitivity();
    m_frameMetadata[streamIdx].ispDigitalGain = iMetadata->getIspDigitalGain();
    m_frameMetadata[streamIdx].sensorAnalogGain = iMetadata->getSensorAnalogGain();
  }

  if (m_captureIsRepeating && (((++m_samplesAtCurrentDuration) > 8) && m_previousSensorTimestampNs)) {

    if (boost::accumulators::rolling_count(m_captureIntervalStats) > 8) {

      int64_t durationToTSDeltaOffset = boost::accumulators::rolling_mean(m_captureIntervalStats) - m_currentCaptureDurationNs;

      // printf("Capture duration % .6f -> interval % .6f (duration-to-interval offset: %ld ns)\n", static_cast<double>(m_currentCaptureDurationNs) / 1000000.0, static_cast<double>(boost::accumulators::rolling_mean(m_captureIntervalStats)) / 1000000.0, durationToTSDeltaOffset);

      int64_t targetDuration = m_targetCaptureIntervalNs - durationToTSDeltaOffset;
      // Clamp the offset to a reasonable small value so that a timestamp discontinuity doesn't cause a massive overshoot
      const int64_t offsetMax = 500000; // 500 microseconds
      int64_t targetOffset = std::min<int64_t>(std::max<int64_t>(targetDuration - m_currentCaptureDurationNs, -offsetMax), offsetMax);
      // printf("durationToTSDeltaOffset %ld targetDuration %ld targetOffset %ld\n", durationToTSDeltaOffset, targetDuration, targetOffset);

      // Perform an adjustment if we're off by at least 1 microsecond
      if (std::abs(targetOffset) > 1000) {
        // Clamp new duration to sensor mode limits
        int64_t newDuration = std::min<int64_t>(std::max<int64_t>(m_currentCaptureDurationNs + (targetOffset / 64), m_captureDurationMinNs), m_captureDurationMaxNs);
        // printf("Capture duration adjust %ld (%ld -> %ld)\n", targetOffset/64, m_currentCaptureDurationNs, newDuration);
        setCaptureDurationNs(newDuration);

        // Start a repeating capture
        if (Argus::interface_cast<Argus::ICaptureSession>(m_captureSession)->repeat(m_captureRequest) != Argus::STATUS_OK)
          die("Failed to update repeat capture request");

      }
    }
  }
  m_previousSensorTimestampNs = m_frameMetadata[0].sensorTimestamp;

  return res;
}

void ArgusCamera::setCaptureDurationNs(uint64_t captureDurationNs) {

  Argus::interface_cast<Argus::ISourceSettings>(m_captureRequest)->setFrameDurationRange(captureDurationNs);
  m_currentCaptureDurationNs = captureDurationNs;
  m_samplesAtCurrentDuration = -64; // Wait long enough to clear out the rolling window of interval stats
}

void ArgusCamera::stop() {
  setRepeatCapture(false);
}

