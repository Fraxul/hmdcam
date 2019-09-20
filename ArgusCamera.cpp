#include "ArgusCamera.h"
#include "ArgusHelpers.h"
#include "GLUtils.h"
#include <GLES2/gl2ext.h>
#include <EGL/eglext.h>
#include <Argus/Argus.h>

PFNEGLCREATESTREAMKHRPROC eglCreateStreamKHR;
PFNEGLDESTROYSTREAMKHRPROC eglDestroyStreamKHR;
PFNEGLSTREAMATTRIBKHRPROC eglStreamAttribKHR;
PFNEGLQUERYSTREAMKHRPROC eglQueryStreamKHR;
PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC eglStreamConsumerGLTextureExternalKHR;
PFNEGLSTREAMCONSUMERACQUIREKHRPROC eglStreamConsumerAcquireKHR;
PFNEGLSTREAMCONSUMERRELEASEKHRPROC eglStreamConsumerReleaseKHR;
PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC eglCreateStreamProducerSurfaceKHR;

PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES;

ArgusCamera::ArgusCamera(EGLDisplay display_, EGLContext context_, unsigned int cameraIndex, unsigned int width, unsigned int height) :
  m_display(display_), m_context(context_),
  m_cameraDevice(NULL), m_captureSession(NULL), m_streamSettings(NULL), m_outputStream(NULL), m_captureRequest(NULL) {
  

  static Argus::CameraProvider* s_cameraProvider = NULL;
  static Argus::ICameraProvider* s_iCameraProvider = NULL;

  if (!s_cameraProvider) {
    s_cameraProvider = Argus::CameraProvider::create();
    s_iCameraProvider = Argus::interface_cast<Argus::ICameraProvider>(s_cameraProvider);
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

    glEGLImageTargetTexture2DOES = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC) eglGetProcAddress("glEGLImageTargetTexture2DOES");
    assert(glEGLImageTargetTexture2DOES);

    eglCreateStreamKHR = (PFNEGLCREATESTREAMKHRPROC) eglGetProcAddress("eglCreateStreamKHR");
    eglDestroyStreamKHR = (PFNEGLDESTROYSTREAMKHRPROC) eglGetProcAddress("eglDestroyStreamKHR");
    eglStreamAttribKHR = (PFNEGLSTREAMATTRIBKHRPROC) eglGetProcAddress("eglStreamAttribKHR");
    eglQueryStreamKHR = (PFNEGLQUERYSTREAMKHRPROC) eglGetProcAddress("eglQueryStreamKHR");
    eglStreamConsumerGLTextureExternalKHR = (PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALKHRPROC) eglGetProcAddress("eglStreamConsumerGLTextureExternalKHR");
    eglStreamConsumerAcquireKHR = (PFNEGLSTREAMCONSUMERACQUIREKHRPROC) eglGetProcAddress("eglStreamConsumerAcquireKHR");
    eglStreamConsumerReleaseKHR = (PFNEGLSTREAMCONSUMERRELEASEKHRPROC) eglGetProcAddress("eglStreamConsumerReleaseKHR");
    eglCreateStreamProducerSurfaceKHR = (PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC) eglGetProcAddress("eglCreateStreamProducerSurfaceKHR");
  }


  // Get the selected camera device and sensor mode.
  m_cameraDevice = ArgusHelpers::getCameraDevice(s_cameraProvider, cameraIndex);
  if (!m_cameraDevice)
      die("Selected camera device is not available");

  ArgusHelpers::printCameraDeviceInfo(m_cameraDevice, "  ");

  const unsigned int sensorModeIndex = 3; // TODO autoselect sensor mode

  Argus::SensorMode* sensorMode = ArgusHelpers::getSensorMode(m_cameraDevice, sensorModeIndex);
  Argus::ISensorMode *iSensorMode = Argus::interface_cast<Argus::ISensorMode>(sensorMode);
  if (!iSensorMode)
      die("Selected sensor mode not available");

  printf("Selected sensor mode:\n");
  ArgusHelpers::printSensorModeInfo(sensorMode, "-- ");

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
  iStreamSettings->setResolution(Argus::Size2D<uint32_t>(width, height));
  iStreamSettings->setEGLDisplay(m_display);

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
  glGenTextures(1, &m_texture);
  glBindTexture(GL_TEXTURE_EXTERNAL_OES, m_texture);
  // Connect the stream consumer. This must be done before starting the capture session, or libargus will return an invalid state error.
  if (!eglStreamConsumerGLTextureExternalKHR(m_display, m_stream))
    die("Unable to connect GL as EGLStream consumer");

  // Set the acquire timeout to infinite.
  eglStreamAttribKHR(m_display, m_stream, EGL_CONSUMER_ACQUIRE_TIMEOUT_USEC_KHR, -1);

  // Start a repeating capture
  if (iCaptureSession->repeat(m_captureRequest) != Argus::STATUS_OK)
    die("Failed to start repeat capture request");
}

ArgusCamera::~ArgusCamera() {
  // Stop the repeating request and signal end of stream to stop the rendering thread.
  Argus::ICaptureSession *iCaptureSession = Argus::interface_cast<Argus::ICaptureSession>(m_captureSession);
  iCaptureSession->stopRepeat();

  // Destroy the output stream.
  m_outputStream.reset();

  // TODO: Shut down Argus after all instances are released
  // g_cameraProvider.reset();

  glDeleteTextures(1, &m_texture);
}

bool ArgusCamera::readFrame() {
  return eglStreamConsumerAcquireKHR(m_display, m_stream);
}

void ArgusCamera::stop() {

}

