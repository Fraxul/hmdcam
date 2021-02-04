#include "rhi/gl/GLCommon.h"

#include <algorithm>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>

void checkGLError(const char* op, const char* file, int line) {
  GLint error = glGetError();
  if (error)
    fprintf(stderr, "after %s (%s:%d) glError (0x%x)\n", op, file, line, error);
}

#ifdef GLATTER_EGL_GLES_3_2
void checkEGLError(const char* op, const char* file, int line) {
  GLint error = eglGetError();
  if (error)
    fprintf(stderr, "after %s (%s:%d) eglError (0x%x)\n", op, file, line, error);
}
#endif


void initGL() {
  printf("OpenGL vendor: %s\n", glGetString(GL_VENDOR));
  printf("OpenGL renderer: %s\n", glGetString(GL_RENDERER));
  printf("Version: %s\n", glGetString(GL_VERSION));
  printf("Shading language version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

/*
  GLenum glewErr = glewInit();
  if (glewErr != GLEW_OK) {
    fprintf(stderr, "GLEW initialization failed: %s", glewGetErrorString(glewErr));
    assert(false && "initGL(): GLEW initialization failed");
  }
*/

}

