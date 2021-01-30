#pragma once

#define GLATTER_EGL_GLES_3_2
#define EGL_EGLEXT_PROTOTYPES
#include <glatter/glatter.h>

#include <stddef.h>
#include <stdint.h>

#ifndef NDEBUG
#define GL(x) x; checkGLError(#x, __FILE__, __LINE__);
#define EGL(x) x; checkEGLError(#x, __FILE__, __LINE__);
#else
#define GL(x) x;
#define EGL(x) x;
#endif

void checkGLError(const char* op, const char* file, int line);
void checkEGLError(const char* op, const char* file, int line);
void initGL();

