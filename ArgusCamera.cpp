#include "ArgusCamera.h"
#include "ArgusHelpers.h"
#include "rhi/gl/GLCommon.h"
#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

ArgusCamera::ArgusCamera(EGLDisplay display_, EGLContext context_, std::vector<unsigned int> cameraIds, unsigned int width, unsigned int height, double framerate) :
  m_display(display_), m_context(context_),
  m_cameraIds(cameraIds),
  m_captureSession(NULL), m_captureRequest(NULL) {

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
    // Select sensor mode. Pick the fastest mode (smallest FrameDurationRange.min) that matches the requested resolution.
    uint64_t bestFrameDurationRangeMin = UINT64_MAX;

    std::vector<Argus::SensorMode*> sensorModes;
    iCameraProperties->getAllSensorModes(&sensorModes);
    for (size_t modeIdx = 0; modeIdx < sensorModes.size(); ++modeIdx) {
      Argus::SensorMode* sensorModeCandidate = sensorModes[modeIdx];
      Argus::ISensorMode *iSensorModeCandidate = Argus::interface_cast<Argus::ISensorMode>(sensorModeCandidate);

      if (!(iSensorModeCandidate->getResolution().width() == width && iSensorModeCandidate->getResolution().height() == height)) {
        continue; // Match failed: wrong resolution
      }

      if (iSensorModeCandidate->getFrameDurationRange().min() < bestFrameDurationRangeMin) {
        bestFrameDurationRangeMin = iSensorModeCandidate->getFrameDurationRange().min();
        sensorMode = sensorModeCandidate;
      }
    }
  }

  if (!sensorMode)
    die("Unable to select a sensor mode matching the requested resolution (%ux%u)", width, height);

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
  m_sensorTimestamps.resize(m_eglStreams.size());

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
  // Set the framerate to the maximum that the sensor mode supports.
  // TODO: The sensor we're working with goes to 120fps, while our display tops out at 90fps.
  // Should probably cap/sync the sensor framerate to display rate to improve noise.
  //iSourceSettings->setFrameDurationRange(iSensorMode->getFrameDurationRange().min());
  iSourceSettings->setFrameDurationRange(/*nanoseconds*/ 1000000000.0 / framerate);

  // Start a repeating capture
  if (iCaptureSession->repeat(m_captureRequest) != Argus::STATUS_OK)
    die("Failed to start repeat capture request");
}

ArgusCamera::~ArgusCamera() {
  for (Argus::OutputStream* outputStream : m_outputStreams)
    outputStream->destroy();
  m_outputStreams.clear();

}

bool ArgusCamera::readFrame() {
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
    m_sensorTimestamps[streamIdx] = iMetadata->getSensorTimestamp();
  }

  return res;
}

void ArgusCamera::stop() {
  // Stop the repeating request and signal end of stream to stop the rendering thread.
  Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(m_captureSession);
  iCaptureSession->stopRepeat();
  iCaptureSession->waitForIdle();
}

