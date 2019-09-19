/*
 * nvgldemo.h
 *
 * Copyright (c) 2007-2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __NVGLDEMO_H
#define __NVGLDEMO_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVGLDEMO_DEFAULT_SURFACE_ID 9000

#define MAX_EGL_STREAM_ATTR 16

//
// OS-dependent functionality
//

// For standard functions, we use macros to allow overrides.
// For now, just use posix.
#define EXIT    exit
#define MALLOC  malloc
#define REALLOC realloc
#define FREE    free
#define MEMSET  memset
#define MEMCPY  memcpy
#define STRLEN  strlen
#define STRCMP  strcmp
#define STRNCMP strncmp
#define STRTOL  strtol
#define STRTOF  strtof
#define STRNCPY strncpy
#define STRSTR  strstr
#define SNPRINTF snprintf
#define POW     (float)pow
#define SQRT    (float)sqrt
#define ISQRT(val) ((float)1.0/(float)sqrt(val))
#define COS     (float)cos
#define SIN     (float)sin
#define ATAN2   (float)atan2
#define PI      (float)M_PI
#define ASSERT  assert

#define NVGLDEMO_EGL_GET_PROC_ADDR(name, fail, type)          \
    do {                                          \
        p##name = (type)eglGetProcAddress(#name);\
        if (!p##name) {                           \
            NvGlDemoLog("%s load fail.\n",#name); \
            goto fail;                         \
        }                                         \
    } while (0)

#define NVGLDEMO_EGL_GET_DISPLAY(nativeDisplay) \
    eglGetDisplay(nativeDisplay)

#define NVGLDEMO_EGL_INITIALIZE(eglDisplay, major, minor) \
        eglInitialize(eglDisplay, major, minor)

#define NVGLDEMO_EGL_QUERY_STRING(eglDisplay, name) \
        eglQueryString(eglDisplay, name)


long long NvGlDemoSysTime(void);
#define SYSTIME (NvGlDemoSysTime)

//
// Window/graphics state
//

// Window System and EGL objects
typedef struct NvGlDemoPlatformState NvGlDemoPlatformState;
typedef struct {
    NativeDisplayType       nativeDisplay;
    NativeWindowType        nativeWindow;
    EGLDisplay              display;
    EGLStreamKHR            stream;
    EGLSurface              surface;
    EGLConfig               config;
    EGLContext              context;
    EGLint                  width;
    EGLint                  height;
    NvGlDemoPlatformState*  platform;
} NvGlDemoState;

extern NvGlDemoState demoState;
extern int           g_ServerID;
extern int           g_ClientID;

int NvGlDemoInitializeEGL(int depthbits, int stencilbits);
int NvGlDemoDisplayInit(void);
void NvGlDemoDisplayTerm(void);
void NvGlDemoShutdown(void);

// Window system specific functions
int
NvGlDemoWindowInit(void);

void
NvGlDemoWindowTerm(void);

EGLBoolean
NvGlDemoPrepareStreamToAttachProducer(void);

void
NvGlDemoEglTerminate(void);

void
NvGlDemoSetDisplayAlpha(float alpha);

EGLBoolean
NvGlDemoSwapInterval(EGLDisplay dpy, EGLint interval);

//
// Command line parsing
//
typedef enum {
    NVGL_DEMO_OPTION_TIMEOUT = 1 << 0
} NvGlDemoOption;

// Parsed options
typedef struct {
    int windowSize[2];                      // Window size
    int windowOffset[2];                    // Window offset
    int windowOffsetValid;                  // Window offset was requested
    int displaySize[2];                     // Display size
    int useCurrentMode;                     // Keeps the current display mode
    int displayRate;                        // Display refresh rate
    int displayLayer;                       // Display layer
    float displayAlpha;                     // Display constant blending alpha
    int msaa;                               // Multi-sampling
    int csaa;                               // Coverage sampling
    int displayNumber;                      // Display output number
    int nFifo;                              // FIFO mode for eglstreams. 0 -> mailbox
} NvGlDemoOptions;

extern NvGlDemoOptions demoOptions;

#ifdef __cplusplus
}
#endif

#endif // __NVGLDEMO_H
