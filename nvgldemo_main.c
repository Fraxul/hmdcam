/*
 * nvgldemo_main.c
 *
 * Copyright (c) 2007-2018, NVIDIA CORPORATION. All rights reserved.
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

//
// This file illustrates how to set up the main rendering context and
//   surface for a GL application.
//

#include "nvgldemo.h"
#include <GLES3/gl3.h>
#include <unistd.h>
#define NvGlDemoLog(msg, ...) fprintf(stderr, msg"\n" , ##__VA_ARGS__)

// Global demo state
NvGlDemoState demoState = {
    (NativeDisplayType)0,  // nativeDisplay
    (NativeWindowType)0,   // nativeWindow
    EGL_NO_DISPLAY,        // display
    EGL_NO_STREAM_KHR,     // stream
    EGL_NO_SURFACE,        // surface
    (EGLConfig)0,          // config
    EGL_NO_CONTEXT,        // context
    0,                     // width
    0,                     // height
    NULL                   // platform
};

// Used with cross-p mode.
// The user has to set these. Not sure if there is a better place to put this.
int g_ServerID = -1;
int g_ClientID = -1;
int g_ServerAllocatedID = -1;

// EGL Device specific variable
static EGLint        devCount = 0;
static EGLDeviceEXT* devList = NULL;
static PFNEGLQUERYDEVICESEXTPROC    peglQueryDevicesEXT = NULL;

// Maximum number of attributes for EGL calls
#define MAX_ATTRIB 31

EGLBoolean NvGlDemoPrepareStreamToAttachProducer(void)
{
    EGLBoolean eglStatus = EGL_FALSE;
    EGLint streamState = EGL_STREAM_STATE_EMPTY_KHR;

    PFNEGLQUERYSTREAMKHRPROC peglQueryStreamKHR = NULL;
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryStreamKHR, fail, PFNEGLQUERYSTREAMKHRPROC);

    // Wait for the consumer to connect to the stream or for failure
    do {
        eglStatus = peglQueryStreamKHR(demoState.display, demoState.stream,
                                       EGL_STREAM_STATE_KHR, &streamState);
        if (!eglStatus) {
            NvGlDemoLog("Producer : Could not query EGL stream state\n");
            goto fail;
        }
    } while ((streamState == EGL_STREAM_STATE_INITIALIZING_NV) ||
              (streamState == EGL_STREAM_STATE_CREATED_KHR));

   // Should now be in CONNECTING state
    if (streamState != EGL_STREAM_STATE_CONNECTING_KHR) {
        NvGlDemoLog("Producer: Stream in bad state\n");
        goto fail;
    }

    return EGL_TRUE;

fail:
    return EGL_FALSE;
}

static void NvGlDemoTermEglDeviceExt(void)
{
    if (devList) {
        FREE(devList);
    }

    devList = NULL;
    devCount = 0;
    peglQueryDevicesEXT = NULL;

    demoState.nativeDisplay = EGL_NO_DISPLAY;
}

EGLContext NvGlDemoCreateShareContext(void) {
  EGLint ctxAttrs[] = {
    EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE
  };

  EGLContext context = eglCreateContext(demoState.display, demoState.config, demoState.context, ctxAttrs);
  if (!context) {
    NvGlDemoLog("NvGlDemoCreateShareContext(): eglCreateContext failed.\n");
  }
  return context;
}

// Start up, initializing native window system and EGL after nvgldemo
//   options have been parsed. (Still need argc/argv for window system
//   options.)
int
NvGlDemoInitializeEGL(int depthbits, int stencilbits)
{
    EGLint cfgAttrs[2*MAX_ATTRIB+1], cfgAttrIndex=0;
    EGLint ctxAttrs[2*MAX_ATTRIB+1], ctxAttrIndex=0;
    EGLint srfAttrs[2*MAX_ATTRIB+1], srfAttrIndex=0;
    const char* extensions;
    EGLConfig* configList = NULL;
    EGLint     configCount;
    EGLBoolean eglStatus;
    GLint max_VP_dims[] = {-1, -1};


    // Initialize display access
    if (!NvGlDemoDisplayInit()) return 0;

    // Obtain the EGL display
    demoState.display = EGL_NO_DISPLAY;
    PFNEGLGETPLATFORMDISPLAYEXTPROC  peglGetPlatformDisplayEXT = NULL;
    NVGLDEMO_EGL_GET_PROC_ADDR(eglGetPlatformDisplayEXT, fail, PFNEGLGETPLATFORMDISPLAYEXTPROC);
    demoState.display = peglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, demoState.nativeDisplay, NULL);

    if (demoState.display == EGL_NO_DISPLAY) {
        NvGlDemoLog("EGL failed to obtain display.\n");
        goto fail;
    }

    // Initialize EGL
    eglStatus = NVGLDEMO_EGL_INITIALIZE(demoState.display, 0, 0);
    if (!eglStatus) {
        NvGlDemoLog("EGL failed to initialize.\n");
        goto fail;
    }

    // Create the window
    if (!NvGlDemoWindowInit()) {
      NvGlDemoLog("NvGlDemoWindowInit() failed\n");
      goto fail;
    }

    // Query EGL extensions
    extensions = NVGLDEMO_EGL_QUERY_STRING(demoState.display, EGL_EXTENSIONS);

    // Bind GL API
    eglBindAPI(EGL_OPENGL_ES_API);

    PFNEGLQUERYSTREAMKHRPROC peglQueryStreamKHR = NULL;
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryStreamKHR, fail, PFNEGLQUERYSTREAMKHRPROC);

    // Request GL version
    cfgAttrs[cfgAttrIndex++] = EGL_RENDERABLE_TYPE;
    cfgAttrs[cfgAttrIndex++] =  EGL_OPENGL_ES3_BIT;
    ctxAttrs[ctxAttrIndex++] = EGL_CONTEXT_CLIENT_VERSION;
    ctxAttrs[ctxAttrIndex++] = 3;

    // Request a minimum of 1 bit each for red, green, blue, and alpha
    // Setting these to anything other than DONT_CARE causes the returned
    //   configs to be sorted with the largest bit counts first.
    cfgAttrs[cfgAttrIndex++] = EGL_RED_SIZE;
    cfgAttrs[cfgAttrIndex++] = 1;
    cfgAttrs[cfgAttrIndex++] = EGL_GREEN_SIZE;
    cfgAttrs[cfgAttrIndex++] = 1;
    cfgAttrs[cfgAttrIndex++] = EGL_BLUE_SIZE;
    cfgAttrs[cfgAttrIndex++] = 1;
    cfgAttrs[cfgAttrIndex++] = EGL_ALPHA_SIZE;
    cfgAttrs[cfgAttrIndex++] = 1;
    cfgAttrs[cfgAttrIndex++] = EGL_SURFACE_TYPE;
    cfgAttrs[cfgAttrIndex++] = EGL_STREAM_BIT_KHR;
    srfAttrs[srfAttrIndex++] = EGL_WIDTH;
    srfAttrs[srfAttrIndex++] = demoOptions.windowSize[0];
    srfAttrs[srfAttrIndex++] = EGL_HEIGHT;
    srfAttrs[srfAttrIndex++] = demoOptions.windowSize[1];

    // If application requires depth or stencil, request them
    if (depthbits) {
        cfgAttrs[cfgAttrIndex++] = EGL_DEPTH_SIZE;
        cfgAttrs[cfgAttrIndex++] = depthbits;
    }
    if (stencilbits) {
        cfgAttrs[cfgAttrIndex++] = EGL_STENCIL_SIZE;
        cfgAttrs[cfgAttrIndex++] = stencilbits;
    }

    // Request antialiasing
    cfgAttrs[cfgAttrIndex++] = EGL_SAMPLES;
    cfgAttrs[cfgAttrIndex++] = demoOptions.msaa;
#ifdef EGL_NV_coverage_sample
    if (STRSTR(extensions, "EGL_NV_coverage_sample")) {
        cfgAttrs[cfgAttrIndex++] = EGL_COVERAGE_SAMPLES_NV;
        cfgAttrs[cfgAttrIndex++] = demoOptions.csaa;
        cfgAttrs[cfgAttrIndex++] = EGL_COVERAGE_BUFFERS_NV;
        cfgAttrs[cfgAttrIndex++] = demoOptions.csaa ? 1 : 0;
    } else
#endif // EGL_NV_coverage_sample
    if (demoOptions.csaa) {
        NvGlDemoLog("Coverage sampling not supported.\n");
        goto fail;
    }

    // Terminate attribute lists
    cfgAttrs[cfgAttrIndex++] = EGL_NONE;
    ctxAttrs[ctxAttrIndex++] = EGL_NONE;
    srfAttrs[srfAttrIndex++] = EGL_NONE;

    // Find out how many configurations suit our needs
    eglStatus = eglChooseConfig(demoState.display, cfgAttrs,
                                NULL, 0, &configCount);
    if (!eglStatus || !configCount) {
        NvGlDemoLog("EGL failed to return any matching configurations.\n");
        goto fail;
    }

    // Allocate room for the list of matching configurations
    configList = (EGLConfig*)MALLOC(configCount * sizeof(EGLConfig));
    if (!configList) {
        NvGlDemoLog("Allocation failure obtaining configuration list.\n");
        goto fail;
    }

    // Obtain the configuration list from EGL
    eglStatus = eglChooseConfig(demoState.display, cfgAttrs,
                                configList, configCount, &configCount);
    if (!eglStatus || !configCount) {
        NvGlDemoLog("EGL failed to populate configuration list.\n");
        goto fail;
    }

    // Select an EGL configuration that matches the native window
    // Currently we just choose the first one, but we could search
    //   the list based on other criteria.
    demoState.config = configList[0];
    FREE(configList);
    configList = 0;

    // Attach surface to stream
    if (!NvGlDemoPrepareStreamToAttachProducer()) {
      NvGlDemoLog("NvGlDemoPrepareStreamToAttachProducer() failed\n");
        goto fail;
    }

    PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC
      peglCreateStreamProducerSurfaceKHR = NULL;
    NVGLDEMO_EGL_GET_PROC_ADDR(eglCreateStreamProducerSurfaceKHR, fail, PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC);
     demoState.surface =
            peglCreateStreamProducerSurfaceKHR(
                demoState.display,
                demoState.config,
                demoState.stream,
                srfAttrs);

    if (demoState.surface == EGL_NO_SURFACE) {
        NvGlDemoLog("EGL couldn't create window surface.\n");
        goto fail;
    }

    // Create an EGL context
    demoState.context =
        eglCreateContext(demoState.display,
                         demoState.config,
                         NULL,
                         ctxAttrs);
    if (!demoState.context) {
        NvGlDemoLog("EGL couldn't create context.\n");
        goto fail;
    }

    // Make the context and surface current for rendering
    eglStatus = eglMakeCurrent(demoState.display,
                               demoState.surface, demoState.surface,
                               demoState.context);
    if (!eglStatus) {
        NvGlDemoLog("EGL couldn't make context/surface current.\n");
        goto fail;
    }

    // Query the EGL surface width and height
    eglStatus =  eglQuerySurface(demoState.display, demoState.surface,
                                 EGL_WIDTH,  &demoState.width)
              && eglQuerySurface(demoState.display, demoState.surface,
                             EGL_HEIGHT, &demoState.height);
    if (!eglStatus) {
        NvGlDemoLog("EGL couldn't get window width/height.\n");
        goto fail;
    }

    // Query the Maximum Viewport width and height
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, max_VP_dims);
    if (max_VP_dims[0] == -1 ||  max_VP_dims[1] == -1) {
        NvGlDemoLog("Couldn't query maximum viewport dimensions.\n");
        goto fail;
    }

    // Check for the Maximum Viewport width and height
    if (demoOptions.windowSize[0] > max_VP_dims[0] ||
        demoOptions.windowSize[1] > max_VP_dims[1]) {
        NvGlDemoLog("Window size exceeds maximum limit of %d x %d.\n",
                    max_VP_dims[0], max_VP_dims[1]);
        goto fail;
    }

    return 1;

    // On failure, clean up partial initialization
fail:
    if (configList) FREE(configList);
    NvGlDemoShutdown();
    return 0;
}

// Shut down, freeing all EGL and native window system resources.
void
NvGlDemoShutdown(void)
{
    EGLBoolean eglStatus;

    // Clear rendering context
    // Note that we need to bind the API to unbind... yick
    if (demoState.display != EGL_NO_DISPLAY) {
        eglBindAPI(EGL_OPENGL_ES_API);
        eglStatus = eglMakeCurrent(demoState.display,
                                   EGL_NO_SURFACE, EGL_NO_SURFACE,
                                   EGL_NO_CONTEXT);
        if (!eglStatus)
            NvGlDemoLog("Error clearing current surfaces/context.\n");
    }

    // Destroy the EGL context
    if (demoState.context != EGL_NO_CONTEXT) {
        eglStatus = eglDestroyContext(demoState.display, demoState.context);
        if (!eglStatus)
            NvGlDemoLog("Error destroying EGL context.\n");
        demoState.context = EGL_NO_CONTEXT;
    }

    // Destroy the EGL surface
    if (demoState.surface != EGL_NO_SURFACE) {
        eglStatus = eglDestroySurface(demoState.display, demoState.surface);
        if (!eglStatus)
            NvGlDemoLog("Error destroying EGL surface.\n");
        demoState.surface = EGL_NO_SURFACE;
    }

#if defined(EGL_KHR_stream_producer_eglsurface)
    // Destroy the EGL stream
    if (demoState.stream != EGL_NO_STREAM_KHR) {
        PFNEGLDESTROYSTREAMKHRPROC
            pEglDestroyStreamKHR;

        pEglDestroyStreamKHR =
            (PFNEGLDESTROYSTREAMKHRPROC)
                eglGetProcAddress("eglDestroyStreamKHR");
        if (pEglDestroyStreamKHR != NULL)
            pEglDestroyStreamKHR(demoState.display, demoState.stream);
        demoState.stream = EGL_NO_STREAM_KHR;
    }
#endif

    // Close the window
    NvGlDemoWindowTerm();

    NvGlDemoEglTerminate();

    // Terminate display access
    NvGlDemoDisplayTerm();

    NvGlDemoTermEglDeviceExt();
}

void
NvGlDemoEglTerminate(void)
{
    EGLBoolean eglStatus;

#if !defined(__INTEGRITY)
    if (g_ServerID != -1) {
        close(g_ServerID);
        g_ServerID = -1;
    }

    if (g_ClientID != -1) {
        close(g_ClientID);
        g_ClientID = -1;
    }
#endif

    // Terminate EGL
    if (demoState.display != EGL_NO_DISPLAY) {
        eglStatus = eglTerminate(demoState.display);
        if (!eglStatus)
            NvGlDemoLog("Error terminating EGL.\n");
        demoState.display = EGL_NO_DISPLAY;
    }

    // Release EGL thread
    eglStatus = eglReleaseThread();
    if (!eglStatus)
        NvGlDemoLog("Error releasing EGL thread.\n");
}

#ifdef EGL_NV_system_time
// Gets the system time in nanoseconds
long long
NvGlDemoSysTime(void)
{
    static PFNEGLGETSYSTEMTIMENVPROC eglGetSystemTimeNV = NULL;
    static int inited = 0;
    static long long nano = 1;
    if(!inited)
    {
        PFNEGLGETSYSTEMTIMEFREQUENCYNVPROC eglGetSystemTimeFrequencyNV =
            (PFNEGLGETSYSTEMTIMEFREQUENCYNVPROC)eglGetProcAddress("eglGetSystemTimeFrequencyNV");
        eglGetSystemTimeNV = (PFNEGLGETSYSTEMTIMENVPROC)eglGetProcAddress("eglGetSystemTimeNV");

        ASSERT(eglGetSystemTimeFrequencyNV && eglGetSystemTimeNV);

        // Compute factor for converting eglGetSystemTimeNV() to nanoseconds
        nano = 1000000000/eglGetSystemTimeFrequencyNV();
        inited = 1;
    }

    return nano*eglGetSystemTimeNV();
}
#endif // EGL_NV_system_time
