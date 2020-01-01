#pragma once
#include <vector>
#include <EGL/egl.h>
#include <Argus/Argus.h>
#include "rhi/gl/RHIEGLStreamSurfaceGL.h"

class ArgusCamera {
public:
  ArgusCamera(EGLDisplay, EGLContext, std::vector<unsigned int> cameraIndices, unsigned int width, unsigned int height, double framerate);
  ~ArgusCamera();

  bool readFrame();

  void stop();

  size_t streamCount() const { return m_eglStreams.size(); }
  RHISurface::ptr rgbTexture(size_t sensorIndex) const { return m_textures[sensorIndex]; }
  unsigned int streamWidth() const { return m_streamWidth; }
  unsigned int streamHeight() const { return m_streamHeight; }

private:
  EGLDisplay m_display;
  EGLContext m_context;

  std::vector<unsigned int> m_cameraIds;
  std::vector<RHIEGLStreamSurfaceGL::ptr> m_textures;
  std::vector<EGLStreamKHR> m_eglStreams;
  unsigned int m_streamWidth, m_streamHeight;

  // Per-sensor objects
  std::vector<Argus::CameraDevice*> m_cameraDevices;
  std::vector<Argus::OutputStream*> m_outputStreams;

  // Session common objects
  Argus::UniqueObj<Argus::CameraProvider> m_cameraProvider;
  Argus::CaptureSession* m_captureSession;
  Argus::Request* m_captureRequest;

  // noncopyable
  ArgusCamera(const ArgusCamera&);
  ArgusCamera& operator=(const ArgusCamera&);
};

