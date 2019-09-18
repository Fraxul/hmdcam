#pragma once
#include <cstddef>
#include <stdio.h>
#include <GLES2/gl2.h>

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

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

// Create a framebuffer object backed by a new RGBA8 texture of the specified dimensions,
// and optionally a depth texture. (pass NULL to outDepthTex to disable depth texture creation)
void createFBO(int width, int height, GLuint* outFBO, GLuint* outColorTex, GLuint* outDepthTex = NULL);

