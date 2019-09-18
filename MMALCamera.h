#pragma once
#include "interface/mmal/mmal.h"
#include <GLES2/gl2.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

#define ENABLE_RGB 1
#define ENABLE_YUV 0

class MMALCamera {
public:
  MMALCamera(EGLDisplay, EGLContext);
  ~MMALCamera();

  void init(unsigned int cameraIndex, unsigned int width, unsigned int height);
  bool readFrame();

  void stop();

#if ENABLE_YUV
  GLuint yTexture() const { return m_yTex; }
  GLuint uTexture() const { return m_uTex; }
  GLuint vTexture() const { return m_vTex; }
#endif
  GLuint rgbTexture() const { return m_rgbTex; }

protected:
  void controlCallback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer);
  void previewOutputCallback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer);
  void do_update_texture(EGLenum target, EGLClientBuffer mm_buf, GLuint *texture, EGLImageKHR *egl_image);

  static void cameraControlCallback_thunk(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer);
  static void previewOutputCallback_thunk(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer);

  EGLDisplay m_eglDisplay;
  EGLContext m_eglContext;

#if ENABLE_YUV
  GLuint m_yTex, m_uTex, m_vTex;
  EGLImageKHR m_yImage, m_uImage, m_vImage;
#endif

#if ENABLE_RGB
  GLuint m_rgbTex;
  EGLImageKHR m_rgbImage;
#endif

  MMAL_COMPONENT_T* m_cameraComponent;

  MMAL_PORT_T *m_previewPort;
  MMAL_POOL_T *m_previewPool;
  MMAL_QUEUE_T *m_previewQueue;

  MMAL_BUFFER_HEADER_T* m_previewBuf;

};
