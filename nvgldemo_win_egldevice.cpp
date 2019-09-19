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

#include <ctype.h>
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "nvgldemo.h"
#define NvGlDemoLog(msg, ...) fprintf(stderr, msg"\n" , ##__VA_ARGS__)
#include "nvgldemo_win_egldevice.h"

#define ARRAY_LEN(_arr) (sizeof(_arr) / sizeof(_arr[0]))

//======================================================================
// Backend initialization
//======================================================================

// Library access
static bool          isOutputInitDone = false;

// EGL Device specific variable
static EGLint        devCount = 0;
static EGLDeviceEXT* devList = NULL;
static struct NvGlOutputDevice *nvGlOutDevLst = NULL;

static PFNEGLQUERYDEVICESEXTPROC         peglQueryDevicesEXT = NULL;
static PFNEGLQUERYDEVICESTRINGEXTPROC    peglQueryDeviceStringEXT = NULL;
static PFNEGLQUERYDEVICEATTRIBEXTPROC    peglQueryDeviceAttribEXT = NULL;
static PFNEGLGETPLATFORMDISPLAYEXTPROC   peglGetPlatformDisplayEXT = NULL;
static PFNEGLGETOUTPUTLAYERSEXTPROC      peglGetOutputLayersEXT = NULL;
static PFNEGLQUERYOUTPUTLAYERSTRINGEXTPROC
peglQueryOutputLayerStringEXT = NULL;
static PFNEGLQUERYOUTPUTLAYERATTRIBEXTPROC
peglQueryOutputLayerAttribEXT = NULL;
static PFNEGLCREATESTREAMKHRPROC         peglCreateStreamKHR = NULL;
static PFNEGLDESTROYSTREAMKHRPROC        peglDestroyStreamKHR = NULL;
static PFNEGLSTREAMCONSUMEROUTPUTEXTPROC peglStreamConsumerOutputEXT = NULL;
static PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC
peglCreateStreamProducerSurfaceKHR = NULL;
static PFNEGLOUTPUTLAYERATTRIBEXTPROC      peglOutputLayerAttribEXT = NULL;

// DRM Device specific variable
static void*         libDRM   = NULL;
static struct NvGlDemoDRMDevice *nvGlDrmDev = NULL;

typedef int (*PFNDRMOPEN)(const char*, const char*);
typedef int (*PFNDRMCLOSE)(int);
typedef drmModeResPtr (*PFNDRMMODEGETRESOURCES)(int);
typedef void (*PFNDRMMODEFREERESOURCES)(drmModeResPtr);
typedef drmModePlaneResPtr (*PFNDRMMODEGETPLANERESOURCES)(int);
typedef void (*PFNDRMMODEFREEPLANERESOURCES)(drmModePlaneResPtr);
typedef drmModeConnectorPtr (*PFNDRMMODEGETCONNECTOR)(int, uint32_t);
typedef void (*PFNDRMMODEFREECONNECTOR)(drmModeConnectorPtr);
typedef drmModeEncoderPtr (*PFNDRMMODEGETENCODER)(int, uint32_t);
typedef void (*PFNDRMMODEFREEENCODER)(drmModeEncoderPtr);
typedef drmModePlanePtr (*PFNDRMMODEGETPLANE)(int, uint32_t);
typedef void (*PFNDRMMODEFREEPLANE)(drmModePlanePtr);
typedef int (*PFNDRMMODESETCRTC)(int, uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t*, int, drmModeModeInfoPtr);
typedef drmModeCrtcPtr (*PFNDRMMODEGETCRTC)(int, uint32_t);
typedef int (*PFNDRMMODESETPLANE)(int, uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, uint32_t, uint32_t);
typedef void (*PFNDRMMODEFREECRTC)(drmModeCrtcPtr);
typedef drmModeAtomicReqPtr (*PFNDRMMODEATOMICALLOC)(void);
typedef int (*PFNDRMMODEATOMICADDPROPERTY)(drmModeAtomicReqPtr, uint32_t, uint32_t, uint64_t);
typedef int (*PFNDRMMODEATOMICCOMMIT)(int, drmModeAtomicReqPtr, uint32_t, void*);
typedef void (*PFNDRMMODEATOMICFREE)(drmModeAtomicReqPtr);
typedef drmModeObjectPropertiesPtr (*PFNDRMMODEOBJECTGETPROPERTIES)(int, uint32_t, uint32_t);
typedef drmModePropertyPtr (*PFNDRMMODEGETPROPERTY)(int, uint32_t);
typedef void (*PFNDRMMODEFREEPROPERTY)(drmModePropertyPtr);
typedef void (*PFNDRMMODEFREEOBJECTPROPERTIES)(drmModeObjectPropertiesPtr);
typedef int (*PFNDRMSETCLIENTCAP)(int, uint64_t, uint64_t);

static PFNDRMOPEN                   pdrmOpen = NULL;
static PFNDRMCLOSE                  pdrmClose = NULL;
static PFNDRMMODEGETRESOURCES       pdrmModeGetResources = NULL;
static PFNDRMMODEFREERESOURCES      pdrmModeFreeResources = NULL;
static PFNDRMMODEGETPLANERESOURCES  pdrmModeGetPlaneResources = NULL;
static PFNDRMMODEFREEPLANERESOURCES pdrmModeFreePlaneResources = NULL;
static PFNDRMMODEGETCONNECTOR       pdrmModeGetConnector = NULL;
static PFNDRMMODEFREECONNECTOR      pdrmModeFreeConnector = NULL;
static PFNDRMMODEGETENCODER         pdrmModeGetEncoder = NULL;
static PFNDRMMODEFREEENCODER        pdrmModeFreeEncoder = NULL;
static PFNDRMMODEGETPLANE           pdrmModeGetPlane = NULL;
static PFNDRMMODEFREEPLANE          pdrmModeFreePlane = NULL;
static PFNDRMMODESETCRTC            pdrmModeSetCrtc = NULL;
static PFNDRMMODEGETCRTC            pdrmModeGetCrtc = NULL;
static PFNDRMMODESETPLANE           pdrmModeSetPlane = NULL;
static PFNDRMMODEFREECRTC           pdrmModeFreeCrtc = NULL;
static PFNDRMMODEATOMICALLOC        pdrmModeAtomicAlloc = NULL;
static PFNDRMMODEATOMICADDPROPERTY  pdrmModeAtomicAddProperty = NULL;
static PFNDRMMODEATOMICCOMMIT       pdrmModeAtomicCommit = NULL;
static PFNDRMMODEATOMICFREE         pdrmModeAtomicFree = NULL;
static PFNDRMMODEOBJECTGETPROPERTIES pdrmModeObjectGetProperties = NULL;
static PFNDRMMODEGETPROPERTY        pdrmModeGetProperty = NULL;
static PFNDRMMODEFREEPROPERTY       pdrmModeFreeProperty = NULL;
static PFNDRMMODEFREEOBJECTPROPERTIES pdrmModeFreeObjectProperties = NULL;
static PFNDRMSETCLIENTCAP           pdrmSetClientCap = NULL;

// Macro to load function pointers
#if !defined(__INTEGRITY)
#define NVGLDEMO_LOAD_DRM_PTR(name, type)               \
    do {                                      \
        p##name = (type)dlsym(libDRM, #name); \
        if (!p##name) {                       \
            NvGlDemoLog("%s load fail.\n",#name); \
            goto NvGlDemoInitDrmDevice_fail;                     \
        }                                     \
    } while (0)
#else
#define NVGLDEMO_LOAD_DRM_PTR(name, type)               \
    p##name = name
#endif

// Extension checking utility
static bool CheckExtension(const char *exts, const char *ext)
{
    int extLen = (int)strlen(ext);
    const char *end = exts + strlen(exts);

    while (exts < end) {
        while (*exts == ' ') {
            exts++;
        }
        int n = strcspn(exts, " ");
        if ((extLen == n) && (strncmp(ext, exts, n) == 0)) {
            return true;
        }
        exts += n;
    }
    return EGL_FALSE;
}

//======================================================================
// EGL Desktop functions
//======================================================================

static bool NvGlDemoInitEglDevice(void)
{
    const char* exts = NULL;
    EGLint n = 0;

    // Get extension string
    exts = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
    if (!exts) {
        NvGlDemoLog("eglQueryString fail.\n");
        goto NvGlDemoInitEglDevice_fail;
    }

    // Check extensions and load functions needed for using outputs
    if (!CheckExtension(exts, "EGL_EXT_device_base") ||
            !CheckExtension(exts, "EGL_EXT_platform_base") ||
            !CheckExtension(exts, "EGL_EXT_platform_device")) {
        NvGlDemoLog("egldevice platform ext is not there.\n");
        goto NvGlDemoInitEglDevice_fail;
    }

    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryDevicesEXT, NvGlDemoInitEglDevice_fail, PFNEGLQUERYDEVICESEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryDeviceStringEXT, NvGlDemoInitEglDevice_fail, PFNEGLQUERYDEVICESTRINGEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryDeviceAttribEXT, NvGlDemoInitEglDevice_fail, PFNEGLQUERYDEVICEATTRIBEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglGetPlatformDisplayEXT, NvGlDemoInitEglDevice_fail, PFNEGLGETPLATFORMDISPLAYEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglGetOutputLayersEXT, NvGlDemoInitEglDevice_fail, PFNEGLGETOUTPUTLAYERSEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryOutputLayerStringEXT, NvGlDemoInitEglDevice_fail, PFNEGLQUERYOUTPUTLAYERSTRINGEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglQueryOutputLayerAttribEXT, NvGlDemoInitEglDevice_fail, PFNEGLQUERYOUTPUTLAYERATTRIBEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglCreateStreamKHR, NvGlDemoInitEglDevice_fail, PFNEGLCREATESTREAMKHRPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglDestroyStreamKHR, NvGlDemoInitEglDevice_fail, PFNEGLDESTROYSTREAMKHRPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglStreamConsumerOutputEXT, NvGlDemoInitEglDevice_fail, PFNEGLSTREAMCONSUMEROUTPUTEXTPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglCreateStreamProducerSurfaceKHR, NvGlDemoInitEglDevice_fail,  PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC);
    NVGLDEMO_EGL_GET_PROC_ADDR(eglOutputLayerAttribEXT, NvGlDemoInitEglDevice_fail,  PFNEGLOUTPUTLAYERATTRIBEXTPROC);

    // Load device list
    if (!peglQueryDevicesEXT(0, NULL, &n) || !n) {
        NvGlDemoLog("peglQueryDevicesEXT fail.\n");
        goto NvGlDemoInitEglDevice_fail;
    }

    nvGlOutDevLst = (struct NvGlOutputDevice *)MALLOC(n*sizeof(struct NvGlOutputDevice));
    if(!nvGlOutDevLst){
        goto NvGlDemoInitEglDevice_fail;
    }

    devList = (EGLDeviceEXT*)MALLOC(n * sizeof(EGLDeviceEXT));
    if (!devList || !peglQueryDevicesEXT(n, devList, &devCount) || !devCount) {
        NvGlDemoLog("peglQueryDevicesEXT fail.\n");
        goto NvGlDemoInitEglDevice_fail;
    }

    // Intial Setup
    NvGlDemoResetEglDevice();

    if(demoState.platform->curDevIndx < devCount) {
        demoState.nativeDisplay = (NativeDisplayType)devList[demoState.platform->curDevIndx];
        NvGlDemoLog("NvGlDemoInitEglDevice: nativeDisplay = %p\n", demoState.nativeDisplay);
        // Success
        return true;
    }

NvGlDemoInitEglDevice_fail:

    NvGlDemoResetModule();

    return false;
}

// Create EGLDevice desktop
static bool NvGlDemoCreateEglDevice(EGLint devIndx)
{
    struct NvGlOutputDevice *devOut = NULL;
    EGLint n = 0;
    const char* exts = NULL;

    if((!nvGlOutDevLst) || (devIndx >= devCount)){
        goto NvGlDemoCreateEglDevice_fail;
    }

    // Set device
    devOut = &nvGlOutDevLst[devIndx];
    devOut->index = devIndx;
    devOut->device = devList[devIndx];
    devOut->eglDpy = demoState.display;

    if ((devOut->eglDpy==EGL_NO_DISPLAY)) {
        NvGlDemoLog("peglGetPlatformDisplayEXT-fail.\n");
        goto NvGlDemoCreateEglDevice_fail;
    }

    // Check for output extension
    exts = eglQueryString(devOut->eglDpy, EGL_EXTENSIONS);
    if (!exts ||
            !CheckExtension(exts, "EGL_EXT_output_base") ||
            !CheckExtension(exts, "EGL_KHR_stream") ||
            !CheckExtension(exts, "EGL_KHR_stream_producer_eglsurface") ||
            !CheckExtension(exts, "EGL_EXT_stream_consumer_egloutput")) {
        NvGlDemoLog("eglstream ext is not there..\n");
        goto NvGlDemoCreateEglDevice_fail;
    }

    // Obtain the total number of available layers and allocate an array of window pointers for them
    if (!peglGetOutputLayersEXT(devOut->eglDpy, NULL, NULL, 0, &n) || !n) {
        NvGlDemoLog("peglGetOutputLayersEXT_fail[%u]\n",n);
        goto NvGlDemoCreateEglDevice_fail;
    }
    devOut->layerList  = (EGLOutputLayerEXT*)MALLOC(n * sizeof(EGLOutputLayerEXT));
    devOut->windowList = (struct NvGlDemoWindowDevice*)MALLOC(n * sizeof(struct NvGlDemoWindowDevice));
    if (devOut->layerList && devOut->windowList) {
        NvGlDemoResetEglDeviceLyrLst(devOut);
        memset(devOut->windowList, 0, (n*sizeof(struct NvGlDemoWindowDevice)));
        memset(devOut->layerList, 0, (n*sizeof(EGLOutputLayerEXT)));
    } else {
        NvGlDemoLog("Failed to allocate list of layers and windows");
        goto NvGlDemoCreateEglDevice_fail;
    }

    devOut->enflag = true;
    return true;

NvGlDemoCreateEglDevice_fail:

    NvGlDemoTermEglDevice();
    return false;
}

// Create the EGL Device surface
static bool NvGlDemoCreateSurfaceBuffer(void)
{
    EGLint layerIndex;
    struct NvGlOutputDevice *outDev = NULL;
    struct NvGlDemoWindowDevice *winDev = NULL;
    EGLint swapInterval = 1;
    EGLint attr[MAX_EGL_STREAM_ATTR * 2 + 1];
    int attrIdx        = 0;

    if (demoOptions.nFifo > 0)
    {
        attr[attrIdx++] = EGL_STREAM_FIFO_LENGTH_KHR;
        attr[attrIdx++] = demoOptions.nFifo;
    }

    attr[attrIdx++] = EGL_NONE;

    if((!nvGlOutDevLst) || (!demoState.platform) || \
            (demoState.platform->curDevIndx >= devCount)  || \
            (nvGlOutDevLst[demoState.platform->curDevIndx].enflag == false)){
        return false;
    }
    outDev = &nvGlOutDevLst[demoState.platform->curDevIndx];

    // Fail if no layers available
    if ((!outDev) || (outDev->layerUsed >= outDev->layerCount) || (!outDev->windowList) || \
            (!outDev->layerList)){
        return false;
    }

    // Try the default
    if ((outDev->layerDefault < outDev->layerCount) && (outDev->windowList[outDev->layerDefault].enflag == false)) {
        outDev->layerIndex = outDev->layerDefault;
    }

    // If that's not available either, find the first unused layer
    else {
        for (layerIndex=0; layerIndex < outDev->layerCount; ++layerIndex) {
            if (outDev->windowList[outDev->layerDefault].enflag == false) {
                break;
            }
        }
        assert(layerIndex < outDev->layerCount);
        outDev->layerIndex = layerIndex;
    }

    outDev->layerUsed++;
    winDev = &outDev->windowList[outDev->layerIndex];

    //Create a stream
    if (demoState.stream == EGL_NO_STREAM_KHR) {
        winDev->stream = peglCreateStreamKHR(outDev->eglDpy, attr);
        demoState.stream = winDev->stream;
    }

    if (demoState.stream == EGL_NO_STREAM_KHR) {
        return false;
    }

    // Connect the output layer to the stream
    if (!peglStreamConsumerOutputEXT(outDev->eglDpy, demoState.stream,
                outDev->layerList[outDev->layerIndex])) {
        return false;
    }

    if (!NvGlDemoSwapInterval(outDev->eglDpy, swapInterval)) {
        return false;
    }

    winDev->index = outDev->layerIndex;
    winDev->enflag = true;

    return true;
}

//Reset EGL Device Layer List
static void NvGlDemoResetEglDeviceLyrLst(struct NvGlOutputDevice *devOut)
{
    int indx;
    for(indx=0;((devOut && devOut->windowList)&&(indx<devOut->layerCount));indx++){
        devOut->windowList[indx].enflag = false;
        devOut->windowList[indx].index = 0;
        devOut->windowList[indx].stream = EGL_NO_STREAM_KHR;
        devOut->windowList[indx].surface = EGL_NO_SURFACE;
    }
    return;
}

// Destroy all EGL Output Devices
static void NvGlDemoResetEglDevice(void)
{
    int indx;
    for(indx=0;indx<devCount;indx++){
        nvGlOutDevLst[indx].enflag = false;
        nvGlOutDevLst[indx].index = 0;
        nvGlOutDevLst[indx].device = 0;
        nvGlOutDevLst[indx].eglDpy = 0;
        nvGlOutDevLst[indx].layerCount = 0;
        nvGlOutDevLst[indx].layerDefault = 0;
        nvGlOutDevLst[indx].layerIndex = 0;
        nvGlOutDevLst[indx].layerUsed = 0;
        NvGlDemoResetEglDeviceLyrLst(&nvGlOutDevLst[indx]);
        nvGlOutDevLst[indx].layerList = NULL;
        nvGlOutDevLst[indx].windowList = NULL;
    }
}

// Free EGL Device stream buffer
static void NvGlDemoTermWinSurface(void)
{
    struct NvGlOutputDevice *outDev = NULL;
    struct NvGlDemoWindowDevice *winDev = NULL;

    if((!nvGlOutDevLst) || (!demoState.platform) || \
            (demoState.platform->curDevIndx >= devCount)  || \
            (nvGlOutDevLst[demoState.platform->curDevIndx].enflag == false)){
        return;
    }
    outDev = &nvGlOutDevLst[demoState.platform->curDevIndx];
    // Fail if no layers available
    if ((!outDev) || (outDev->layerUsed > outDev->layerCount) || (!outDev->windowList) || \
            (!outDev->layerList) || (outDev->layerIndex >= outDev->layerCount)){
        return;
    }
    winDev = &outDev->windowList[outDev->layerIndex];
    if((!winDev) || (winDev->enflag == false)){
        NvGlDemoLog("NvGlDemoTermWinSurface-fail\n");
        return;
    }

    if (winDev->stream != EGL_NO_STREAM_KHR && demoState.stream != EGL_NO_STREAM_KHR) {
        (void)peglDestroyStreamKHR(outDev->eglDpy, winDev->stream);
        demoState.stream = EGL_NO_STREAM_KHR;
    }

    outDev->windowList[winDev->index].enflag = false;
    outDev->windowList[winDev->index].index = 0;
    outDev->windowList[winDev->index].stream = EGL_NO_STREAM_KHR;
    outDev->windowList[winDev->index].surface = EGL_NO_SURFACE;
    outDev->layerUsed--;
    outDev->eglDpy = EGL_NO_DISPLAY;

    demoState.platform->curDevIndx = 0;
    return;
}

// Terminate Egl Device
static void NvGlDemoTermEglDevice(void)
{
    if(nvGlOutDevLst)
    {
        int indx;
        for(indx=0;indx<devCount;indx++) {
            if(nvGlOutDevLst[indx].layerList){
                FREE(nvGlOutDevLst[indx].layerList);
            }
            if(nvGlOutDevLst[indx].windowList) {
                FREE(nvGlOutDevLst[indx].windowList);
            }
            demoState.nativeDisplay = NULL;
            nvGlOutDevLst[indx].eglDpy = EGL_NO_DISPLAY;
        }
        if(devList) {
            FREE(devList);
        }
        FREE(nvGlOutDevLst);
        devList = NULL;
        nvGlOutDevLst = NULL;
        devCount = 0;
        NvGlDemoResetEglDeviceFnPtr();
    }
    return;
}

// Reset all EGL Device Function ptr
static void NvGlDemoResetEglDeviceFnPtr(void)
{
    peglQueryDevicesEXT = NULL;
    peglQueryDeviceStringEXT = NULL;
    peglQueryDeviceAttribEXT = NULL;
    peglGetPlatformDisplayEXT = NULL;
    peglGetOutputLayersEXT = NULL;
    peglQueryOutputLayerStringEXT = NULL;
    peglQueryOutputLayerAttribEXT = NULL;
    peglCreateStreamKHR = NULL;
    peglDestroyStreamKHR = NULL;
    peglStreamConsumerOutputEXT = NULL;
    peglCreateStreamProducerSurfaceKHR = NULL;
    peglOutputLayerAttribEXT = NULL;
}

//======================================================================
// DRM Desktop functions
//======================================================================

// Load the EGL and DRM libraries if available
static bool NvGlDemoInitDrmDevice(void)
{
#if !defined(__INTEGRITY)
    // Open DRM library
    libDRM = dlopen("libdrm.so.2", RTLD_LAZY);
    if (!libDRM) {
        NvGlDemoLog("dlopen-libdrm.so.2 fail.\n");
        return false;
    }
#endif

    // Get DRM functions
    NVGLDEMO_LOAD_DRM_PTR(drmOpen, PFNDRMOPEN);
    NVGLDEMO_LOAD_DRM_PTR(drmClose, PFNDRMCLOSE);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetResources, PFNDRMMODEGETRESOURCES);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeResources, PFNDRMMODEFREERESOURCES);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetPlaneResources, PFNDRMMODEGETPLANERESOURCES);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreePlaneResources, PFNDRMMODEFREEPLANERESOURCES);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetConnector, PFNDRMMODEGETCONNECTOR);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeConnector, PFNDRMMODEFREECONNECTOR);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetEncoder, PFNDRMMODEGETENCODER);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeEncoder, PFNDRMMODEFREEENCODER);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetPlane, PFNDRMMODEGETPLANE);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreePlane, PFNDRMMODEFREEPLANE);
    NVGLDEMO_LOAD_DRM_PTR(drmModeSetCrtc, PFNDRMMODESETCRTC);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetCrtc, PFNDRMMODEGETCRTC);
    NVGLDEMO_LOAD_DRM_PTR(drmModeSetPlane, PFNDRMMODESETPLANE);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeCrtc, PFNDRMMODEFREECRTC);
    NVGLDEMO_LOAD_DRM_PTR(drmModeAtomicAlloc, PFNDRMMODEATOMICALLOC);
    NVGLDEMO_LOAD_DRM_PTR(drmModeAtomicAddProperty, PFNDRMMODEATOMICADDPROPERTY);
    NVGLDEMO_LOAD_DRM_PTR(drmModeAtomicCommit, PFNDRMMODEATOMICCOMMIT);
    NVGLDEMO_LOAD_DRM_PTR(drmModeAtomicFree, PFNDRMMODEATOMICFREE);
    NVGLDEMO_LOAD_DRM_PTR(drmModeObjectGetProperties, PFNDRMMODEOBJECTGETPROPERTIES);
    NVGLDEMO_LOAD_DRM_PTR(drmModeGetProperty, PFNDRMMODEGETPROPERTY);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeProperty, PFNDRMMODEFREEPROPERTY);
    NVGLDEMO_LOAD_DRM_PTR(drmModeFreeObjectProperties, PFNDRMMODEFREEOBJECTPROPERTIES);
    NVGLDEMO_LOAD_DRM_PTR(drmSetClientCap, PFNDRMSETCLIENTCAP);

    nvGlDrmDev =
        (struct NvGlDemoDRMDevice*)MALLOC(sizeof(struct NvGlDemoDRMDevice));
    if (!nvGlDrmDev) {
        NvGlDemoLog("Could not allocate Drm Device specific storage memory.\n");
        goto NvGlDemoInitDrmDevice_fail;
    }
    NvGlDemoResetDrmDevice();

    // Success
    return true;

NvGlDemoInitDrmDevice_fail:

    NvGlDemoTermDrmDevice();
    NvGlDemoLog("NvGlDemoCreateDrmDevice fail.\n");
    return false;
}

//Return the plane type for the specified objectID
static int GetDrmPlaneType(int drmFd, uint32_t objectID)
{
    uint32_t i;
    int j;
    int found = 0;
    uint64_t value = 0;
    int planeType = DRM_PLANE_TYPE_OVERLAY;

    drmModeObjectPropertiesPtr pModeObjectProperties =
        pdrmModeObjectGetProperties(drmFd, objectID, DRM_MODE_OBJECT_PLANE);

    for (i = 0; i < pModeObjectProperties->count_props; i++) {
       drmModePropertyPtr pProperty =
           pdrmModeGetProperty(drmFd, pModeObjectProperties->props[i]);

       if (pProperty == NULL) {
           NvGlDemoLog("Unable to query property.\n");
       }

       if(STRCMP("type", pProperty->name) == 0) {
           value = pModeObjectProperties->prop_values[i];
           found = 1;
           for (j = 0; j < pProperty->count_enums; j++) {
               if (value == (pProperty->enums[j]).value) {
                   if (STRCMP( "Primary", (pProperty->enums[j]).name) == 0) {
                       planeType = DRM_PLANE_TYPE_PRIMARY;
                   } else if (STRCMP( "Overlay", (pProperty->enums[j]).name) == 0) {
                       planeType = DRM_PLANE_TYPE_OVERLAY;
                   } else {
                       planeType = DRM_PLANE_TYPE_CURSOR;
                   }
               }
           }
       }

       pdrmModeFreeProperty(pProperty);

       if (found)
           break;
    }

    pdrmModeFreeObjectProperties(pModeObjectProperties);

    if (!found) {
       NvGlDemoLog("Unable to find value for property \'type.\'\n");
    }

    return planeType;
}

// Create DRM/EGLDevice desktop
static bool NvGlDemoCreateDrmDevice( EGLint devIndx )
{
    struct NvGlOutputDevice *devOut = NULL;
    EGLOutputLayerEXT tempLayer;
    int i = 0, j = 0, n = 0, layerIndex = 0;

    EGLAttrib layerAttrib[3] = {
        0,
        0,
        EGL_NONE
    };

    devOut = &nvGlOutDevLst[devIndx];

    // Open DRM file and query resources
    if ((nvGlDrmDev->drmName = peglQueryDeviceStringEXT(devList[devIndx], EGL_DRM_DEVICE_FILE_EXT)) == NULL) {
        NvGlDemoLog("EGL_DRM_DEVICE_FILE_EXT fail\n");
        return false;
    }
    if ((nvGlDrmDev->fd = pdrmOpen(nvGlDrmDev->drmName, NULL)) == 0) {
        NvGlDemoLog("%s open fail\n",nvGlDrmDev->drmName);
        return false;
    }

    if (!(pdrmSetClientCap(nvGlDrmDev->fd, DRM_CLIENT_CAP_ATOMIC, 1) == 0)) {
        NvGlDemoLog("DRM_CLIENT_CAP_ATOMIC not available.\n");
        return false;
    }

    if (!(pdrmSetClientCap(nvGlDrmDev->fd, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1) == 0)) {
        NvGlDemoLog("DRM_CLIENT_CAP_UNIVERSAL_PLANES not available.\n");
        return false;
    }

    if ((nvGlDrmDev->res = pdrmModeGetResources(nvGlDrmDev->fd)) == NULL) {
        NvGlDemoLog("pdrmModeGetResources fail\n");
        return false;
    }
    if ((nvGlDrmDev->planes = pdrmModeGetPlaneResources(nvGlDrmDev->fd)) == NULL) {
        NvGlDemoLog("pdrmModeGetPlaneResources fail\n");
        return false;
    }
    // Validate connector, if requested
    if (nvGlDrmDev->connDefault >= nvGlDrmDev->res->count_connectors) {
        NvGlDemoLog("con def != max con\n");
        return false;
    }

    // Allocate info arrays for DRM state
    nvGlDrmDev->connInfo = (NvGlDemoDRMConn*)malloc(nvGlDrmDev->res->count_connectors * sizeof(NvGlDemoDRMConn));
    nvGlDrmDev->crtcInfo = (NvGlDemoDRMCrtc*)malloc(nvGlDrmDev->res->count_crtcs * sizeof(NvGlDemoDRMCrtc));
    nvGlDrmDev->planeInfo = (NvGlDemoDRMPlane*)malloc(nvGlDrmDev->planes->count_planes * sizeof(NvGlDemoDRMPlane));
    if (!nvGlDrmDev->connInfo || !nvGlDrmDev->crtcInfo || !nvGlDrmDev->planeInfo) {
        NvGlDemoLog("Drm Res Alloc fail\n");
        return false;
    }
    memset(nvGlDrmDev->connInfo, 0, nvGlDrmDev->res->count_connectors * sizeof(NvGlDemoDRMConn));
    memset(nvGlDrmDev->crtcInfo, 0, nvGlDrmDev->res->count_crtcs * sizeof(NvGlDemoDRMCrtc));
    memset(nvGlDrmDev->planeInfo, 0, nvGlDrmDev->planes->count_planes * sizeof(NvGlDemoDRMPlane));

    // Parse connector info
    for (i=0; i<nvGlDrmDev->res->count_connectors; ++i) {

        // Start with no crtc assigned
        nvGlDrmDev->connInfo[i].crtcMapping = -1;

        // Skip if not connector
        drmModeConnector* conn = pdrmModeGetConnector(nvGlDrmDev->fd, nvGlDrmDev->res->connectors[i]);
        if (!conn || (conn->connection != DRM_MODE_CONNECTED)) {
            if (conn) {
                // Free the connector info
                pdrmModeFreeConnector(conn);
            }
            continue;
        }

        // If the connector has no modes available, try to use the current
        // mode.  Show a warning if the user didn't specifically request that.
        if (conn->count_modes <= 0 && (!demoOptions.useCurrentMode)) {
            NvGlDemoLog("Warning: No valid modes found for connector, "
                        "using -currentmode\n");
            demoOptions.useCurrentMode = 1;
        }

        // If we don't already have a default, use this one
        if (nvGlDrmDev->connDefault < 0) {
            nvGlDrmDev->connDefault = i;
        }

        // Mark as valid
        nvGlDrmDev->connInfo[i].valid = true;

        // Find the possible crtcs
        for (j=0; j<conn->count_encoders; ++j) {
            drmModeEncoder* enc = pdrmModeGetEncoder(nvGlDrmDev->fd, conn->encoders[j]);
            nvGlDrmDev->connInfo[i].crtcMask |= enc->possible_crtcs;
            pdrmModeFreeEncoder(enc);
        }
        // Free the connector info
        pdrmModeFreeConnector(conn);
    }

    // Parse crtc info
    for (i=0; i<nvGlDrmDev->res->count_crtcs; ++i) {
        nvGlDrmDev->crtcInfo[i].layer = -1;
    }

    // Parse plane info
    for (i=0; i<(int)nvGlDrmDev->planes->count_planes; ++i) {
        drmModePlane* plane = pdrmModeGetPlane(nvGlDrmDev->fd, nvGlDrmDev->planes->planes[i]);
        nvGlDrmDev->planeInfo[i].layer = -1;
        nvGlDrmDev->planeInfo[i].crtcMask = plane->possible_crtcs;
        pdrmModeFreePlane(plane);
        nvGlDrmDev->planeInfo[i].planeType = GetDrmPlaneType(nvGlDrmDev->fd, nvGlDrmDev->planes->planes[i]);
    }

    // Map layers to crtc
    layerAttrib[0] = EGL_DRM_CRTC_EXT;
    for (i=0; i<nvGlDrmDev->res->count_crtcs; ++i) {
        layerAttrib[1] = nvGlDrmDev->res->crtcs[i];
        if (peglGetOutputLayersEXT(devOut->eglDpy, layerAttrib, &tempLayer, 1, &n) && n > 0) {
            devOut->layerList[layerIndex] = tempLayer;
            devOut->layerCount++;
            nvGlDrmDev->crtcInfo[i].layer = layerIndex++;
        }
    }

    // Map layers to planes
    layerAttrib[0] = EGL_DRM_PLANE_EXT;
    for (i=0; i<(int)nvGlDrmDev->planes->count_planes; ++i) {
        layerAttrib[1] = nvGlDrmDev->planes->planes[i];
        if (peglGetOutputLayersEXT(devOut->eglDpy, layerAttrib, &tempLayer, 1, &n) && n > 0) {
            devOut->layerList[layerIndex] = tempLayer;
            devOut->layerCount++;
            nvGlDrmDev->planeInfo[i].layer = layerIndex++;
        }
    }

    if (!devOut->layerCount) {
        NvGlDemoLog("Layer count is 0.\n");
        return false;
    }

    return true;
}

/*
 * Query the properties for the specified object, and populate the IDs
 * in the given table.
 */
static bool AssignPropertyIDs(int drmFd,
                              uint32_t objectID,
                              uint32_t objectType,
                              struct PropertyIDAddress *table,
                              size_t tableLen)
{
    uint32_t i;
    drmModeObjectPropertiesPtr pModeObjectProperties =
        pdrmModeObjectGetProperties(drmFd, objectID, objectType);

    if (pModeObjectProperties == NULL) {
        NvGlDemoLog("Unable to query mode object properties.\n");
        return false;
    }

    for (i = 0; i < pModeObjectProperties->count_props; i++) {

        uint32_t j;
        drmModePropertyPtr pProperty =
            pdrmModeGetProperty(drmFd, pModeObjectProperties->props[i]);

        if (pProperty == NULL) {
            NvGlDemoLog("Unable to query property.\n");
            return false;
        }

        for (j = 0; j < tableLen; j++) {
            if (strcmp(table[j].name, pProperty->name) == 0) {
                *(table[j].ptr) = pProperty->prop_id;
                break;
            }
        }

        pdrmModeFreeProperty(pProperty);
    }

    pdrmModeFreeObjectProperties(pModeObjectProperties);

    for (i = 0; i < tableLen; i++) {
        if (*(table[i].ptr) == 0) {
            NvGlDemoLog("Unable to find property ID for \'%s\'.\n", table[i].name);
        }
    }

    return true;
}

// Set output mode
static bool NvGlDemoSetDrmOutputMode( void )
{
    int offsetX = 0;
    int offsetY = 0;
    unsigned int sizeX = 0;
    unsigned int sizeY = 0;
    unsigned int alpha = 255;

    int crtcIndex = -1;
    int i = 0;
    unsigned int planeIndex = ~0;
    unsigned int currPlaneIndex = 0;

    drmModeConnector* conn = NULL;
    drmModeEncoder* enc = NULL;
    struct NvGlOutputDevice *outDev = NULL;
    drmModeCrtcPtr currMode = NULL;

    unsigned int modeSize = 0;
    int modeIndex = -1;
    unsigned int modeX = 0;
    unsigned int modeY = 0;
    int foundMatchingDisplayRate = 1;

    // Parse global plane alpha
    if(demoOptions.displayAlpha < 0.0 || demoOptions.displayAlpha > 1.0) {
        //If unspecified or out of range, default to 1.0
        NvGlDemoLog("Alpha value specified for constant blending is not in range [0, 1]. Using alpha 1.0.\n");
        demoOptions.displayAlpha = 1.0;
    }
    alpha = (unsigned int)(demoOptions.displayAlpha * 255);

    offsetX = demoOptions.windowOffset[0];
    offsetY = demoOptions.windowOffset[1];
    sizeX = demoOptions.windowSize[0];
    sizeY = demoOptions.windowSize[1];

    nvGlDrmDev->curConnIndx = demoState.platform->curConnIndx;
    // If a specific screen was requested, use it
    if ((nvGlDrmDev->curConnIndx >= nvGlDrmDev->res->count_connectors) ||
            !nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].valid) {
        NvGlDemoLog("Display output %d is not available, try using another display using option <-dispno>.\n",nvGlDrmDev->curConnIndx);
        goto NvGlDemoSetDrmOutputMode_fail;
    }

    // Get the current state of the connector
    conn = pdrmModeGetConnector(nvGlDrmDev->fd, nvGlDrmDev->res->connectors[nvGlDrmDev->curConnIndx]);
    if (!conn) {
        NvGlDemoLog("pdrmModeGetConnector-fail\n");
        goto NvGlDemoSetDrmOutputMode_fail;
    }
    enc = pdrmModeGetEncoder(nvGlDrmDev->fd, conn->encoder_id);
    if (enc) {
        for (i=0; i<nvGlDrmDev->res->count_crtcs; ++i) {
            if (nvGlDrmDev->res->crtcs[i] == enc->crtc_id) {
                nvGlDrmDev->currCrtcIndx = i;
            }
        }
        pdrmModeFreeEncoder(enc);
    }

    if (nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].crtcMapping >= 0) {
        crtcIndex = nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].crtcMapping;
        assert(crtcIndex == nvGlDrmDev->currCrtcIndx);
    } else if (nvGlDrmDev->currCrtcIndx >= 0) {
        crtcIndex = nvGlDrmDev->currCrtcIndx;
        assert(!nvGlDrmDev->crtcInfo[crtcIndex].mapped);
    } else {
        for (crtcIndex=0; crtcIndex<nvGlDrmDev->res->count_crtcs; ++crtcIndex) {
            if (!nvGlDrmDev->crtcInfo[crtcIndex].mapped &&
                    (nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].crtcMask & (1 << crtcIndex))) {
                break;
            }
        }
        if (crtcIndex == nvGlDrmDev->res->count_crtcs) {
            goto NvGlDemoSetDrmOutputMode_fail;
        }
    }

    // Set the CRTC if we haven't already done it
    if (!nvGlDrmDev->crtcInfo[crtcIndex].mapped) {
        if (demoOptions.displaySize[0]) {

            // Check whether the choosen mode is supported or not
            for (i=0; i<conn->count_modes; ++i) {
                drmModeModeInfoPtr mode = conn->modes + i;
                if (mode->hdisplay == demoOptions.displaySize[0]
                    && mode->vdisplay == demoOptions.displaySize[1]) {
                    modeIndex = i;
                    modeX = (unsigned int)mode->hdisplay;
                    modeY = (unsigned int)mode->vdisplay;
                    if (demoOptions.displayRate) {
                        if (mode->vrefresh == (unsigned int)demoOptions.displayRate) {
                            foundMatchingDisplayRate = 1;
                            break;
                        } else {
                            foundMatchingDisplayRate = 0;
                            continue;
                        }
                    }
                    break;
                }
            }
            if (!modeX || !modeY) {
                NvGlDemoLog("Unsupported Displaysize.\n");
                goto NvGlDemoSetDrmOutputMode_fail;
            }
            if (!foundMatchingDisplayRate) {
                NvGlDemoLog("Specified Refresh rate is not Supported with Specified Display size.\n");
                goto NvGlDemoSetDrmOutputMode_fail;
            }
        } else if (demoOptions.useCurrentMode) {
            if (demoOptions.displayRate) {
                NvGlDemoLog("Refresh Rate should not be specified with Current Mode Parameter.\n");
                goto NvGlDemoSetDrmOutputMode_fail;
            }
            //check modeset
            if ((currMode = (drmModeCrtcPtr)pdrmModeGetCrtc(nvGlDrmDev->fd,
                       nvGlDrmDev->res->crtcs[crtcIndex])) != NULL) {
                modeIndex = -1;
            }
        } else {
            if (demoOptions.displayRate) {
                NvGlDemoLog("Refresh Rate should not be specified alone.\n");
                goto NvGlDemoSetDrmOutputMode_fail;
            }
            // Choose the preferred mode if it's set,
            // Or else choose the largest supported mode
            for (i=0; i<conn->count_modes; ++i) {
                drmModeModeInfoPtr mode = conn->modes + i;
                if (mode->type & DRM_MODE_TYPE_PREFERRED) {
                    modeIndex = i;
                    modeX = (unsigned int)mode->hdisplay;
                    modeY = (unsigned int)mode->vdisplay;
                    break;
                }
                unsigned int size = (unsigned int)mode->hdisplay
                    * (unsigned int)mode->vdisplay;
                if (size > modeSize) {
                    modeIndex = i;
                    modeSize = size;
                    modeX = (unsigned int)mode->hdisplay;
                    modeY = (unsigned int)mode->vdisplay;
                }
            }
        }

        // Set the mode
        if (modeIndex >= 0) {
            if (pdrmModeSetCrtc(nvGlDrmDev->fd, nvGlDrmDev->res->crtcs[crtcIndex], -1,
                        0, 0, &nvGlDrmDev->res->connectors[nvGlDrmDev->curConnIndx], 1,
                        conn->modes + modeIndex)) {
                NvGlDemoLog("pdrmModeSetCrtc-fail setting crtc mode\n");
                goto NvGlDemoSetDrmOutputMode_fail;
            }
        }
        if ((currMode = (drmModeCrtcPtr) pdrmModeGetCrtc(nvGlDrmDev->fd,
            nvGlDrmDev->res->crtcs[crtcIndex])) != NULL) {
            NvGlDemoLog("Demo Mode: %d x %d @ %d\n",
                currMode->mode.hdisplay, currMode->mode.vdisplay,
                currMode->mode.vrefresh);
        } else {
            NvGlDemoLog("Failed to get current mode.\n");
        }

        // Mark connector/crtc as initialized
        nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].crtcMapping = crtcIndex;
        nvGlDrmDev->crtcInfo[crtcIndex].modeX = modeX;
        nvGlDrmDev->crtcInfo[crtcIndex].modeY = modeY;
        nvGlDrmDev->crtcInfo[crtcIndex].mapped = true;
    }

    // If a size wasn't specified, use the whole screen
    if (!sizeX || !sizeY) {
        assert(!sizeX && !sizeY && !offsetX && !offsetY);
        assert(currMode);
        sizeX = currMode->mode.hdisplay;
        sizeY = currMode->mode.vdisplay;
        //sizeX = nvGlDrmDev->crtcInfo[crtcIndex].modeX;
        //sizeY = nvGlDrmDev->crtcInfo[crtcIndex].modeY;
        demoOptions.windowSize[0] = sizeX;
        demoOptions.windowSize[1] = sizeY;
    }

    /* Ideally, only the plane interfaces should be used when universal planes is enabled.
     * There should be less "if crtc, if plane" type conditions in the code when
     * universal planes is supported.
     * To keep the code simple,fail nvgldemo init completely if universal planes and atomics
     * is not working. Everything is done the atomic+universal_planes way.
     */
    for (planeIndex=0; planeIndex<nvGlDrmDev->planes->count_planes; ++planeIndex) {
        if (!nvGlDrmDev->planeInfo[planeIndex].used &&
                (nvGlDrmDev->planeInfo[planeIndex].crtcMask & (1 << crtcIndex)) &&
                (currPlaneIndex++ == (unsigned int)demoOptions.displayLayer)) {
            NvGlDemoLog("Atomic_request\n");
            drmModeAtomicReqPtr pAtomic;
            int ret;
            const uint32_t flags = 0;

            pAtomic = pdrmModeAtomicAlloc();
            if (pAtomic == NULL) {
                NvGlDemoLog("Failed to allocate the property set\n");
                goto NvGlDemoSetDrmOutputMode_fail;
            }

            struct PropertyIDAddress planeTable[] = {
                { "alpha",   &nvGlDrmDev->currPlaneAlphaPropID       },
            };

            if(!AssignPropertyIDs(nvGlDrmDev->fd, nvGlDrmDev->planes->planes[planeIndex] ,
                              DRM_MODE_OBJECT_PLANE, planeTable, ARRAY_LEN(planeTable))) {
                NvGlDemoLog("Failed to assign property IDs\n");
                goto NvGlDemoSetDrmOutputMode_fail;
            }

            pdrmModeAtomicAddProperty(pAtomic, nvGlDrmDev->planes->planes[planeIndex],
                                      nvGlDrmDev->currPlaneAlphaPropID, alpha);

            ret = pdrmModeAtomicCommit(nvGlDrmDev->fd, pAtomic, flags, NULL /* user_data */);

            pdrmModeAtomicFree(pAtomic);

            if (ret != 0) {
                NvGlDemoLog("Failed to commit properties. Error code: %d\n", ret);
                goto NvGlDemoSetDrmOutputMode_fail;
            }

            if (pdrmModeSetPlane(nvGlDrmDev->fd, nvGlDrmDev->planes->planes[planeIndex],
                        nvGlDrmDev->res->crtcs[crtcIndex], -1, 0,
                        offsetX, offsetY, sizeX, sizeY,
                        0, 0, sizeX << 16, sizeY << 16)) {
                NvGlDemoLog("pdrmModeSetPlane-fail\n");
                goto NvGlDemoSetDrmOutputMode_fail;
            }
            break;
        }
    }
    if (planeIndex == nvGlDrmDev->planes->count_planes) {
        NvGlDemoLog("ERROR: Layer ID %d is not valid on display %d.\n",
                     demoOptions.displayLayer,
                     demoOptions.displayNumber);
        NvGlDemoLog("Range of available Layer IDs: [0, %d]",
                    currPlaneIndex - 1);
        goto NvGlDemoSetDrmLayer_fail;
    }
    nvGlDrmDev->currPlaneIndx = planeIndex;
    nvGlDrmDev->isPlane = true;

    outDev = &nvGlOutDevLst[demoState.platform->curDevIndx];

    if(nvGlDrmDev->planeInfo[planeIndex].planeType == DRM_PLANE_TYPE_PRIMARY) {
        outDev->layerDefault = nvGlDrmDev->crtcInfo[crtcIndex].layer;
        nvGlDrmDev->crtcInfo[crtcIndex].used = true;
    } else {
        outDev->layerDefault = nvGlDrmDev->planeInfo[planeIndex].layer;
        nvGlDrmDev->planeInfo[planeIndex].used = true;
    }

    if (conn) {
        pdrmModeFreeConnector(conn);
    }

    if (currMode) {
        pdrmModeFreeCrtc(currMode);
    }

    return true;

NvGlDemoSetDrmOutputMode_fail:

    if (conn != NULL) {

        NvGlDemoLog("List of available display modes\n");
        for (i=0; i<conn->count_modes; ++i) {
            drmModeModeInfoPtr mode = conn->modes + i;
            NvGlDemoLog("%d x %d @ %d\n", mode->hdisplay, mode->vdisplay, mode->vrefresh);
        }
    }

NvGlDemoSetDrmLayer_fail:

    // Clean up and return
    if (conn) {
        pdrmModeFreeConnector(conn);
    }

    if (currMode) {
        pdrmModeFreeCrtc(currMode);
    }

    return false;

}

// Reset DRM Device
static void NvGlDemoResetDrmDevice(void)
{
    if(nvGlDrmDev)
    {
        nvGlDrmDev->fd = 0;
        nvGlDrmDev->drmName = NULL;
        nvGlDrmDev->res = NULL;
        nvGlDrmDev->planes = NULL;
        nvGlDrmDev->connDefault = -1;
        nvGlDrmDev->isPlane = false;
        nvGlDrmDev->curConnIndx = -1;
        nvGlDrmDev->currCrtcIndx = -1;
        nvGlDrmDev->currPlaneIndx = -1;
        nvGlDrmDev->connInfo = NULL;
        nvGlDrmDev->crtcInfo = NULL;
        nvGlDrmDev->planeInfo = NULL;
    }
    return;
}

//Reset DRM Subdriver connection status
static void NvGlDemoResetDrmConnection(void)
{
    int offsetX = 0;
    int offsetY = 0;
    unsigned int sizeX = 0;
    unsigned int sizeY = 0;

    offsetX = demoOptions.windowOffset[0];
    offsetY = demoOptions.windowOffset[1];
    sizeX = demoOptions.windowSize[0];
    sizeY = demoOptions.windowSize[1];

    if(nvGlDrmDev && demoState.platform ) {
        if((nvGlDrmDev->connInfo) && (nvGlDrmDev->curConnIndx != -1)) {
            nvGlDrmDev->connInfo[nvGlDrmDev->curConnIndx].crtcMapping = -1;
        }
        // Mark plane as in unuse
        if((nvGlDrmDev->isPlane) && (nvGlDrmDev->planeInfo) && (nvGlDrmDev->currPlaneIndx != -1) && (nvGlDrmDev->currCrtcIndx != -1)) {
            if (pdrmModeSetPlane(nvGlDrmDev->fd, nvGlDrmDev->planes->planes[nvGlDrmDev->currPlaneIndx],
                        nvGlDrmDev->res->crtcs[nvGlDrmDev->currCrtcIndx], 0, 0,
                        offsetX, offsetY, sizeX, sizeY,
                        0, 0, sizeX << 16, sizeY << 16)) {
                NvGlDemoLog("pdrmModeSetPlane-fail\n");
            }
            nvGlDrmDev->planeInfo[nvGlDrmDev->currPlaneIndx].used = false;
        } else if((nvGlDrmDev->crtcInfo) && (nvGlDrmDev->currCrtcIndx != -1) && (nvGlDrmDev->curConnIndx != -1)) {
            if (pdrmModeSetCrtc(nvGlDrmDev->fd,
                        nvGlDrmDev->res->crtcs[nvGlDrmDev->currCrtcIndx], 0,
                        offsetX, offsetY,
                        &nvGlDrmDev->res->connectors[nvGlDrmDev->curConnIndx],
                        1, NULL)) {
                NvGlDemoLog("pdrmModeSetCrtc-fail\n");
            }
            nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].modeX = 0;
            nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].modeY = 0;
            nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].mapped = false;
            nvGlDrmDev->crtcInfo[nvGlDrmDev->currCrtcIndx].used = false;
        }
        demoState.platform->curConnIndx = 0;
    }
    return;
}

// Terminate Drm Device
static void NvGlDemoTermDrmDevice(void)
{
    if(nvGlDrmDev)
    {
        if(nvGlDrmDev->connInfo) {
            FREE(nvGlDrmDev->connInfo);
        }
        if(nvGlDrmDev->crtcInfo) {
            FREE(nvGlDrmDev->crtcInfo);
        }
        if(nvGlDrmDev->planeInfo) {
            FREE(nvGlDrmDev->planeInfo);
        }
        if (nvGlDrmDev->planes) {
            pdrmModeFreePlaneResources(nvGlDrmDev->planes);
        }
        if (nvGlDrmDev->res) {
            pdrmModeFreeResources(nvGlDrmDev->res);
        }
        if (pdrmClose(nvGlDrmDev->fd)) {
            NvGlDemoLog("drmClose failed\n");
        }
#if !defined(__INTEGRITY)
        if(libDRM) {
            dlclose(libDRM);
            libDRM = NULL;
        }
#endif
        FREE(nvGlDrmDev);
        NvGlDemoResetDrmDeviceFnPtr();
        nvGlDrmDev = NULL;
    }
    return;
}

// Reset all Drm Device Function ptr
static void NvGlDemoResetDrmDeviceFnPtr(void)
{
    pdrmOpen = NULL;
    pdrmClose = NULL;
    pdrmModeGetResources = NULL;
    pdrmModeFreeResources = NULL;
    pdrmModeGetPlaneResources = NULL;
    pdrmModeFreePlaneResources = NULL;
    pdrmModeGetConnector = NULL;
    pdrmModeFreeConnector = NULL;
    pdrmModeGetEncoder = NULL;
    pdrmModeFreeEncoder = NULL;
    pdrmModeGetPlane = NULL;
    pdrmModeFreePlane = NULL;
    pdrmModeSetCrtc = NULL;
    pdrmModeGetCrtc = NULL;
    pdrmModeSetPlane = NULL;
    pdrmModeFreeCrtc = NULL;
    pdrmModeAtomicAlloc = NULL;
    pdrmModeAtomicAddProperty = NULL;
    pdrmModeAtomicCommit = NULL;
    pdrmModeAtomicFree = NULL;
    pdrmModeObjectGetProperties = NULL;
    pdrmModeGetProperty = NULL;
    pdrmModeFreeProperty = NULL;
    pdrmModeFreeObjectProperties = NULL;
    pdrmSetClientCap = NULL;
}

//======================================================================
// Nvgldemo Display functions
//======================================================================

// Initialize access to the display system
int NvGlDemoDisplayInit(void)
{
    // Only try once
    if (isOutputInitDone) {
        return 1;
    }

    // Allocate a structure for the platform-specific state
    demoState.platform =
        (NvGlDemoPlatformState*)MALLOC(sizeof(NvGlDemoPlatformState));
    if (!demoState.platform) {
        NvGlDemoLog("Could not allocate platform specific storage memory.\n");
        goto NvGlDemoDisplayInit_fail;
    }

    demoState.platform->curDevIndx = 0;
    demoState.platform->curConnIndx = demoOptions.displayNumber;

    if(NvGlDemoInitEglDevice()) {
        if(NvGlDemoInitDrmDevice()) {
            // Output functions are available
            isOutputInitDone = true;
            // Success
            return 1;
        }
    }

NvGlDemoDisplayInit_fail:

    NvGlDemoResetModule();

    return 0;
}

// Terminate access to the display system
void NvGlDemoDisplayTerm(void)
{
    if (!isOutputInitDone) {
        NvGlDemoLog("Display_init not yet done[%d].\n",isOutputInitDone);
        return;
    }
    // End Device Setup
    NvGlDemoTermDrmDevice();
    NvGlDemoTermEglDevice();
    // Reset Module
    NvGlDemoResetModule();
}

static void NvGlDemoResetModule(void)
{
    if(demoState.platform) {
        FREE(demoState.platform);
    }

    if(nvGlOutDevLst) {
        FREE(nvGlOutDevLst);
    }

    if(devList) {
        FREE(devList);
    }

    if(nvGlDrmDev) {
        FREE(nvGlDrmDev);
    }

#if !defined(__INTEGRITY)
    if(libDRM) {
        dlclose(libDRM);
    }
#endif

    demoState.platform = NULL;
    demoState.nativeDisplay = NULL;
    nvGlOutDevLst = NULL;
    devList = NULL;
    nvGlDrmDev = NULL;
#if !defined(__INTEGRITY)
    libDRM   = NULL;
#endif

    // Reset module global variables
    isOutputInitDone = false;
    devCount = 0;

    NvGlDemoResetEglDeviceFnPtr();
    NvGlDemoResetDrmDeviceFnPtr();
    return;
}

//======================================================================
// Nvgldemo Window functions
//======================================================================

// Window creation
int NvGlDemoWindowInit(void)
{
    if (!isOutputInitDone) {
        NvGlDemoLog("Display_init not yet done[%d].\n",isOutputInitDone);
        return 0;
    }
    // Create the EGL Device and DRM Device
    if(NvGlDemoCreateEglDevice(demoState.platform->curDevIndx)){
        if(!NvGlDemoCreateDrmDevice(demoState.platform->curDevIndx)){
            NvGlDemoLog("NvGlDemoCreateDrmDevice() failed\n");
            goto NvGlDemoWindowInit_fail;
        }
    }else {
        NvGlDemoLog("NvGlDemoCreateEglDevice() failed\n");
        goto NvGlDemoWindowInit_fail;
    }
    // Make the Output requirement for Devices
    if(NvGlDemoSetDrmOutputMode()){
        if(!NvGlDemoCreateSurfaceBuffer()){
            NvGlDemoLog("NvGlDemoCreateSurfaceBuffer failed\n");
            goto NvGlDemoWindowInit_fail;
        }
    }else {
        NvGlDemoLog("NvGlDemoSetDrmOutputMode failed\n");
        goto NvGlDemoWindowInit_fail;
    }

    return 1;

NvGlDemoWindowInit_fail:

    // Clean up and return
    NvGlDemoWindowTerm();
    NvGlDemoResetModule();
    return 0;
}

// Close the window
void NvGlDemoWindowTerm(void)
{
    if (!isOutputInitDone) {
        NvGlDemoLog("Display_init not yet done[%d].\n",isOutputInitDone);
        return;
    }
    NvGlDemoResetDrmConnection();
    NvGlDemoTermWinSurface();
    return;
}


EGLBoolean NvGlDemoSwapInterval(EGLDisplay dpy, EGLint interval)
{
    struct NvGlOutputDevice *outDev = NULL;
    EGLAttrib swapInterval = 1;
    char *configStr = NULL;

    if((!nvGlOutDevLst) || (!demoState.platform) ||
        (demoState.platform->curDevIndx >= devCount) ||
        (nvGlOutDevLst[demoState.platform->curDevIndx].enflag == false) ||
        (!peglOutputLayerAttribEXT)) {
        return EGL_FALSE;
    }
    outDev = &nvGlOutDevLst[demoState.platform->curDevIndx];
    // Fail if no layers available
    if ((!outDev) || (outDev->layerUsed > outDev->layerCount) || (!outDev->windowList) ||
        (!outDev->layerList)) {
        return EGL_FALSE;
    }

    if ((outDev->layerIndex >= outDev->layerCount)) {
        NvGlDemoLog("NvGlDemoSwapInterval_fail[Layer -%d]\n",outDev->layerIndex);
        return EGL_FALSE;
    }

    // To allow the interval to be overridden by an environment variable exactly the same way like a normal window system.
    configStr = getenv("__GL_SYNC_TO_VBLANK");

    if(!configStr)
        configStr = getenv("NV_SWAPINTERVAL");

    // Environment variable is higher priority than runtime setting
    if (configStr) {
        swapInterval = (EGLAttrib)strtol(configStr, NULL, 10);
    } else {
        swapInterval = (EGLint)interval;
    }

    if (!peglOutputLayerAttribEXT(outDev->eglDpy,
            outDev->layerList[outDev->layerIndex],
            EGL_SWAP_INTERVAL_EXT,
            swapInterval)) {
        NvGlDemoLog("peglOutputLayerAttribEXT_fail[%p %ld]\n",outDev->layerList[outDev->layerIndex],swapInterval);
        return EGL_FALSE;
    }

    return EGL_TRUE;
}

void NvGlDemoSetDisplayAlpha(float alpha)
{
    unsigned int newAlpha = (unsigned int)(alpha * 255);
    if (newAlpha > 255) {
        NvGlDemoLog("Alpha value specified for constant blending in not in specified range [0,1]. Using alpha 1.0\n");
        newAlpha = 255;
    }

    drmModeAtomicReqPtr pAtomic;
    const uint32_t flags = DRM_MODE_ATOMIC_NONBLOCK;

    pAtomic = pdrmModeAtomicAlloc();
    if (pAtomic == NULL) {
        NvGlDemoLog("Failed to allocate the property set\n");
        return;
    }

    pdrmModeAtomicAddProperty(pAtomic, nvGlDrmDev->planes->planes[nvGlDrmDev->currPlaneIndx],
                              nvGlDrmDev->currPlaneAlphaPropID, newAlpha);

    int ret = pdrmModeAtomicCommit(nvGlDrmDev->fd, pAtomic, flags, NULL /* user_data */);

    pdrmModeAtomicFree(pAtomic);

    if (ret != 0) {
        NvGlDemoLog("Failed to commit properties. Error code: %d\n", ret);
    }
}

