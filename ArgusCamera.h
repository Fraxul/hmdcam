#pragma once
#include <GLES2/gl2.h>
#include <EGL/egl.h>
#include <Argus/Argus.h>

#define NUM_BUFFERS 10

class ArgusCamera {
public:
  ArgusCamera(EGLDisplay, EGLContext, unsigned int cameraIndex, unsigned int width, unsigned int height);
  ~ArgusCamera();

  bool readFrame();

  void stop();

  GLuint rgbTexture() const { return m_texture; }

private:
  EGLDisplay m_display;
  EGLContext m_context;

  GLuint m_texture;
  EGLStreamKHR m_stream;

  Argus::CameraDevice* m_cameraDevice;
  Argus::CaptureSession* m_captureSession;
  Argus::OutputStreamSettings* m_streamSettings;
  Argus::UniqueObj<Argus::OutputStream> m_outputStream;
  Argus::Request* m_captureRequest;


  // noncopyable
  ArgusCamera(const ArgusCamera&);
  ArgusCamera& operator=(const ArgusCamera&);
};

