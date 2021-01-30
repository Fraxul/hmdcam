/* Copyright (c) 2014 - 2019 NVIDIA Corporation.  All rights reserved.
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

#ifndef __NVGLDEMO_WIN_EGLDEVICE_H
#define __NVGLDEMO_WIN_EGLDEVICE_H

#include <EGL/egl.h>
#include <EGL/eglext.h>

#include <xf86drm.h>
#include <xf86drmMode.h>

#include <stdbool.h>

#ifndef DRM_PLANE_TYPE_OVERLAY
#define DRM_PLANE_TYPE_OVERLAY 0
#endif

#ifndef DRM_PLANE_TYPE_PRIMARY
#define DRM_PLANE_TYPE_PRIMARY 1
#endif

#ifndef DRM_PLANE_TYPE_CURSOR
#define DRM_PLANE_TYPE_CURSOR  2
#endif

// Platform-specific state info
struct NvGlDemoPlatformState
{
    // Input - Device Instance index
    int      curDevIndx;
    // Input - Connector Index
    int      curConnIndx;
};


// EGLOutputLayer window List
struct NvGlDemoWindowDevice
{
    bool enflag;
    EGLint                  index;
    EGLStreamKHR            stream;
    EGLSurface              surface;
};

// EGLOutputDevice
struct NvGlOutputDevice
{
    bool                             enflag;
    EGLint                           index;
    EGLDeviceEXT                     device;
    EGLDisplay                       eglDpy;
    EGLint                           layerCount;
    EGLint                           layerDefault;
    EGLint                           layerIndex;
    EGLint                           layerUsed;
    EGLOutputLayerEXT*               layerList;
    struct NvGlDemoWindowDevice*     windowList;
};

// Parsed DRM info structures
typedef struct {
    bool             valid;
    unsigned int     crtcMask;
    int              crtcMapping;
} NvGlDemoDRMConn;

typedef struct {
    EGLint       layer;
    unsigned int modeX;
    unsigned int modeY;
    bool         mapped;
    bool         used;
} NvGlDemoDRMCrtc;

typedef struct {
    EGLint           layer;
    unsigned int     crtcMask;
    bool             used;
    int              planeType;
} NvGlDemoDRMPlane;

// DRM+EGLDesktop desktop class
struct NvGlDemoDRMDevice
{
    int                fd;
    const char*        drmName;
    drmModeRes*        res;
    drmModePlaneRes*   planes;

    int                connDefault;
    bool               isPlane;
    int                curConnIndx;
    int                currCrtcIndx;
    int                currPlaneIndx;

    unsigned int       currPlaneAlphaPropID;

    NvGlDemoDRMConn*   connInfo;
    NvGlDemoDRMCrtc*   crtcInfo;
    NvGlDemoDRMPlane*  planeInfo;
};

struct PropertyIDAddress {
    const char*  name;
    uint32_t*    ptr;
};

// EGL Device internal api
static bool NvGlDemoInitEglDevice(void);
static bool NvGlDemoCreateEglDevice(EGLint devIndx);
static bool NvGlDemoCreateSurfaceBuffer(void);
static void NvGlDemoResetEglDeviceLyrLst(struct NvGlOutputDevice *devOut);
static void NvGlDemoResetEglDevice(void);
static void NvGlDemoTermWinSurface(void);
static void NvGlDemoTermEglDevice(void);
static void NvGlDemoResetEglDeviceFnPtr(void);

// DRM Device internal api
static bool NvGlDemoInitDrmDevice(void);
static bool NvGlDemoCreateDrmDevice( EGLint devIndx );
static bool NvGlDemoSetDrmOutputMode( void );
static void NvGlDemoResetDrmDevice(void);
static void NvGlDemoResetDrmConnection(void);
static void NvGlDemoTermDrmDevice(void);
static void NvGlDemoResetDrmDeviceFnPtr(void);

// Module internal api
static void NvGlDemoResetModule(void);

#endif // __NVGLDEMO_WIN_EGLDEVICE_H

