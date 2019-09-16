#pragma once
#include <GLES2/gl2.h>

#ifndef NDEBUG
#define GL(x) x; checkGLError(#x, __FILE__, __LINE__);
#define EGL(x) x; checkEGLError(#x, __FILE__, __LINE__);
#else
#define GL(x) x;
#define EGL(x) x;
#endif

void checkGLError(const char* op, const char* file, int line);
void checkEGLError(const char* op, const char* file, int line);

GLuint compileShader(const char* vertexSource, const char* fragmentSource);
