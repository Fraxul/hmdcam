#pragma once
#include <GLES2/gl2.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

class ArgusCamera {
public:
  ArgusCamera(EGLDisplay, EGLContext);
  ~ArgusCamera();

  void init(unsigned int cameraIndex, unsigned int width, unsigned int height);
  bool readFrame();

  void stop();

  GLuint rgbTexture() const { return 0; }

private:
  // noncopyable
  ArgusCamera(const ArgusCamera&);
  ArgusCamera& operator=(const ArgusCamera&);
};

