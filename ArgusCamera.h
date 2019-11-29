#pragma once
#include <EGL/egl.h>
#include <Argus/Argus.h>
#include "rhi/gl/RHIEGLStreamSurfaceGL.h"

#define NUM_BUFFERS 10

class ArgusCamera {
public:
  ArgusCamera(EGLDisplay, EGLContext, unsigned int cameraIndex, unsigned int width, unsigned int height);
  ~ArgusCamera();

  bool readFrame();

  void stop();

  RHISurface::ptr rgbTexture() const { return m_texture; }
  unsigned int streamWidth() const { return m_streamWidth; }
  unsigned int streamHeight() const { return m_streamHeight; }


private:
  EGLDisplay m_display;
  EGLContext m_context;

  RHIEGLStreamSurfaceGL::ptr m_texture;
  EGLStreamKHR m_stream;
  unsigned int m_streamWidth, m_streamHeight;

  Argus::CameraDevice* m_cameraDevice;
  Argus::CaptureSession* m_captureSession;
  Argus::OutputStreamSettings* m_streamSettings;
  Argus::UniqueObj<Argus::OutputStream> m_outputStream;
  Argus::Request* m_captureRequest;


  // noncopyable
  ArgusCamera(const ArgusCamera&);
  ArgusCamera& operator=(const ArgusCamera&);
};

