#include "ArgusCamera.h"
#include "ArgusHelpers.h"
#include "rhi/gl/GLCommon.h"
#include <Argus/Argus.h>

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

ArgusCamera::ArgusCamera(EGLDisplay display_, EGLContext context_, unsigned int cameraIndex, unsigned int width, unsigned int height) :
  m_display(display_), m_context(context_),
  m_cameraDevice(NULL), m_captureSession(NULL), m_streamSettings(NULL), m_outputStream(NULL), m_captureRequest(NULL) {
  

  static Argus::UniqueObj<Argus::CameraProvider> s_cameraProvider;
  static Argus::ICameraProvider* s_iCameraProvider = NULL;

  if (!s_cameraProvider) {
    s_cameraProvider.reset(Argus::CameraProvider::create());
    s_iCameraProvider = Argus::interface_cast<Argus::ICameraProvider>(s_cameraProvider.get());
    if (!s_iCameraProvider) {
      die("Failed to get ICameraProvider interface");
    }
    printf("Argus Version: %s\n", s_iCameraProvider->getVersion().c_str());

    {
      std::vector<Argus::CameraDevice*> devices;
      s_iCameraProvider->getCameraDevices(&devices);
      printf("Devices (%zu):\n", devices.size());
      for (size_t deviceIdx = 0; deviceIdx < devices.size(); ++deviceIdx) {
        printf("[%zu] %p\n", deviceIdx, devices[deviceIdx]);
      }
    }

  }

  // Get the selected camera device and sensor mode.
  m_cameraDevice = ArgusHelpers::getCameraDevice(s_cameraProvider.get(), cameraIndex);
  if (!m_cameraDevice)
      die("Selected camera device is not available");

  ArgusHelpers::printCameraDeviceInfo(m_cameraDevice, "  ");

  const unsigned int sensorModeIndex = 4; // TODO autoselect sensor mode based on width/height params

  Argus::SensorMode* sensorMode = ArgusHelpers::getSensorMode(m_cameraDevice, sensorModeIndex);
  Argus::ISensorMode *iSensorMode = Argus::interface_cast<Argus::ISensorMode>(sensorMode);
  if (!iSensorMode)
      die("Selected sensor mode not available");

  printf("Selected sensor mode:\n");
  ArgusHelpers::printSensorModeInfo(sensorMode, "-- ");
  m_streamWidth = iSensorMode->getResolution().width();
  m_streamHeight = iSensorMode->getResolution().height();

  // Create the capture session using the specified device and get its interfaces.
  m_captureSession = s_iCameraProvider->createCaptureSession(m_cameraDevice);
  Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(m_captureSession);
  Argus::IEventProvider *iEventProvider = Argus::interface_cast<Argus::IEventProvider>(m_captureSession);
  if (!iCaptureSession || !iEventProvider)
      die("Failed to create CaptureSession");

  // Create the OutputStreamSettings object for an EGL OutputStream
  m_streamSettings = iCaptureSession->createOutputStreamSettings(Argus::STREAM_TYPE_EGL);
  Argus::IEGLOutputStreamSettings *iStreamSettings =
      Argus::interface_cast<Argus::IEGLOutputStreamSettings>(m_streamSettings);
  if (!iStreamSettings)
      die("Failed to create OutputStreamSettings");

  // Configure the OutputStream to use the EGLImage BufferType.
  iStreamSettings->setPixelFormat(Argus::PIXEL_FMT_YCbCr_420_888);
  iStreamSettings->setResolution(Argus::Size2D<uint32_t>(m_streamWidth, m_streamHeight));
  iStreamSettings->setEGLDisplay(m_display);
  iStreamSettings->setMode(Argus::EGL_STREAM_MODE_MAILBOX);

  // Create the OutputStream.
  m_outputStream.reset(iCaptureSession->createOutputStream(m_streamSettings));
  Argus::IEGLOutputStream *iEGLOutputStream = Argus::interface_cast<Argus::IEGLOutputStream>(m_outputStream);
  if (!iEGLOutputStream)
      die("Failed to create EGLOutputStream");
  m_stream = iEGLOutputStream->getEGLStream();


  // Create capture request, set the sensor mode, and enable the output stream.
  m_captureRequest = iCaptureSession->createRequest();
  Argus::IRequest *iRequest = Argus::interface_cast<Argus::IRequest>(m_captureRequest);
  if (!iRequest)
      die("Failed to create Request");
  iRequest->enableOutputStream(m_outputStream.get());
  Argus::ISourceSettings *iSourceSettings = Argus::interface_cast<Argus::ISourceSettings>(m_captureRequest);
  if (!iSourceSettings)
      die("Failed to get source settings request interface");
  iSourceSettings->setSensorMode(sensorMode);
  // Set the framerate to the maximum that the sensor mode supports.
  // TODO: The sensor we're working with goes to 120fps, while our display tops out at 75fps.
  // Should probably cap/sync the sensor framerate to display rate to improve noise.
  iSourceSettings->setFrameDurationRange(iSensorMode->getFrameDurationRange().min());

  // GL texture which will be associated with EGL images in readFrame()
  m_texture = new RHIEGLStreamSurfaceGL(m_streamWidth, m_streamHeight, kSurfaceFormat_RGBA8);

  glBindTexture(GL_TEXTURE_EXTERNAL_OES, m_texture->glId());
  // Connect the stream consumer. This must be done before starting the capture session, or libargus will return an invalid state error.
  if (!eglStreamConsumerGLTextureExternalKHR(m_display, m_stream))
    die("Unable to connect GL as EGLStream consumer");
  glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

  // Set the acquire timeout to infinite.
  eglStreamAttribKHR(m_display, m_stream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, -1);

  // Start a repeating capture
  if (iCaptureSession->repeat(m_captureRequest) != Argus::STATUS_OK)
    die("Failed to start repeat capture request");
}

ArgusCamera::~ArgusCamera() {
  // Destroy the output stream.
  m_outputStream.reset();

  // TODO: Shut down Argus after all instances are released
  // g_cameraProvider.reset();

  m_texture.reset();
}

bool ArgusCamera::readFrame() {
  return eglStreamConsumerAcquireKHR(m_display, m_stream);
}

void ArgusCamera::stop() {
  // Stop the repeating request and signal end of stream to stop the rendering thread.
  Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(m_captureSession);
  iCaptureSession->stopRepeat();
  iCaptureSession->waitForIdle();
}

