#pragma once
#include <cstddef>
#include <stdio.h>
#include "rhi/gl/GLCommon.h"

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

GLuint compileShader(const char* vertexSource, const char* fragmentSource);

// Create a framebuffer object backed by a new RGBA8 texture of the specified dimensions,
// and optionally a depth texture. (pass NULL to outDepthTex to disable depth texture creation)
void createFBO(int width, int height, GLuint* outFBO, GLuint* outColorTex, GLuint* outDepthTex = NULL);

