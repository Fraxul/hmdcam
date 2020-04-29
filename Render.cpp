#include "Render.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/gl/GLCommon.h"
#include "rhi/gl/RHIEGLSurfaceRenderTargetGL.h"

#include "nvgldemo.h"

#include "openhmd/openhmd.h"

#include "NvEncSession.h"
#include "liveMedia.hh"
#include "BasicUsageEnvironment.hh"
#include "BufferRingSource.h"
#include "H264VideoNvEncSessionServerMediaSubsession.h"
#include <cuda.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <sys/time.h>

#define CAMERA_INVERTED 1 // 0 = upright, 1 = camera rotated 180 degrees. (90 degree rotation is not supported)

RHIRenderTarget::ptr windowRenderTarget;

FxAtomicString ksNDCQuadUniformBlock("NDCQuadUniformBlock");
FxAtomicString ksNDCClippedQuadUniformBlock("NDCClippedQuadUniformBlock");
FxAtomicString ksSolidQuadUniformBlock("SolidQuadUniformBlock");

RHIRenderPipeline::ptr camTexturedQuadPipeline;
RHIRenderPipeline::ptr camOverlayPipeline;
RHIRenderPipeline::ptr camOverlayStereoPipeline;
RHIRenderPipeline::ptr camOverlayStereoUndistortPipeline;
RHIRenderPipeline::ptr camUndistortMaskPipeline;
RHIRenderPipeline::ptr camGreyscalePipeline;
RHIRenderPipeline::ptr camGreyscaleUndistortPipeline;
RHIRenderPipeline::ptr solidQuadPipeline;

RHISurface::ptr disabledMaskTex;

struct ViveDistortionUniformBlock {
  glm::vec4 coeffs[3];
  glm::vec4 center;
  float undistort_r2_cutoff;
  float aspect_x_over_y;
  float grow_for_undistort;
  float pad4;
};
FxAtomicString ksViveDistortionUniformBlock("ViveDistortionUniformBlock");

RHIRenderPipeline::ptr viveDistortionPipeline;

FxAtomicString ksImageTex("imageTex");
FxAtomicString ksLeftCameraTex("leftCameraTex");
FxAtomicString ksRightCameraTex("rightCameraTex");
FxAtomicString ksLeftDistortionMap("leftDistortionMap");
FxAtomicString ksRightDistortionMap("rightDistortionMap");
FxAtomicString ksOverlayTex("overlayTex");
FxAtomicString ksLeftOverlayTex("leftOverlayTex");
FxAtomicString ksRightOverlayTex("rightOverlayTex");
FxAtomicString ksDistortionMap("distortionMap");
FxAtomicString ksMaskTex("maskTex");

// per-eye render targets (pre distortion)
RHISurface::ptr eyeTex[2];
RHIRenderTarget::ptr eyeRT[2];
// per-eye distortion parameter buffers
RHIBuffer::ptr viveDistortionParams[2];

// HMD info/state
ohmd_context* hmdContext = NULL;
ohmd_device* hmdDevice = NULL;
unsigned int hmd_width, hmd_height;
unsigned int eye_width, eye_height;
bool rotate_screen = false;
glm::mat4 eyeProjection[2];
glm::mat4 eyeView[2];

float stereoSeparationScale = 2.0f; // TODO: why? some of the Vive projection math is still wrong...

NvGlDemoOptions demoOptions;

// CUDA
CUdevice cudaDevice;
CUcontext cudaContext;

// Streaming server / NvEnc state

NvEncSession* nvencSession;

TaskScheduler* rtspScheduler;
BasicUsageEnvironment* rtspEnv;
RTSPServer* rtspServer;
ServerMediaSession* rtspMediaSession;
std::string rtspURL;
uint64_t rtspRenderIntervalNs = 33333333; // 30fps
//uint64_t rtspRenderIntervalNs = 66666667; // 15fps
//uint64_t rtspRenderIntervalNs = 11169814; // 89.527fps

// -----------

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

void* rtspServerThreadEntryPoint(void* arg) {
  printf("Starting RTSP server event loop\n");
  EGLContext eglCtx = NvGlDemoCreateShareContext();
  if (!eglCtx) {
    die("rtspServerThreadEntryPoint: unable to create EGL share context\n");
  }

  bool res = eglMakeCurrent(demoState.display, EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx);
  if (!res) {
    die("rtspServerThreadEntryPoint: eglMakeCurrent() failed\n");
  }

  cuCtxSetCurrent(cudaContext);

  rtspEnv->taskScheduler().doEventLoop();

  eglDestroyContext(demoState.display, eglCtx);

  return NULL;
}

bool RenderInit() {
  memset(&demoOptions, 0, sizeof(demoOptions));

  demoOptions.displayAlpha = 1.0;
  demoOptions.nFifo = 1;

  // Use the current mode and the entire screen
  demoOptions.useCurrentMode = 1;
  demoOptions.windowSize[0] = 0;
  demoOptions.windowSize[1] = 0;

  if (!NvGlDemoInitializeEGL(0, 0)) {
    printf("NvGlDemoInitializeEGL() failed\n");
    return false;
  }

  printf("%s\n", glGetString(GL_RENDERER));
  printf("%s\n", glGetString(GL_VERSION));
  printf("%s\n", glGetString(GL_EXTENSIONS));

  // CUDA init
  {
    cuInit(0);

    cuDeviceGet(&cudaDevice, 0);
    char devName[512];
    cuDeviceGetName(devName, 511, cudaDevice);
    devName[511] = '\0';
    printf("CUDA device: %s\n", devName);

    cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice);
    cuCtxSetCurrent(cudaContext);
  }

  initRHIGL();

  RHIEGLSurfaceRenderTargetGL::ptr wrt(new RHIEGLSurfaceRenderTargetGL(demoState.display, demoState.surface));
  wrt->platformSetUpdatedWindowDimensions(demoState.width, demoState.height);
  windowRenderTarget = wrt;

  // Set up shared resources

  {
    RHIShaderDescriptor desc(
    "shaders/ndcQuadXf.vtx.glsl",
    "shaders/camTexturedQuad.frag.glsl",
    ndcQuadVertexLayout);
    desc.setFlag("CAMERA_INVERTED", (bool) CAMERA_INVERTED);

    camTexturedQuadPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  {
    RHIShaderDescriptor desc(
      "shaders/ndcQuadXf.vtx.glsl",
      "shaders/camOverlay.frag.glsl",
      ndcQuadVertexLayout);
    desc.setFlag("CAMERA_INVERTED", (bool) CAMERA_INVERTED);
    camOverlayPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  {
    RHIShaderDescriptor desc(
      "shaders/ndcQuadXf.vtx.glsl",
      "shaders/camOverlayStereo.frag.glsl",
      ndcQuadVertexLayout);
    desc.setFlag("CAMERA_INVERTED", (bool) CAMERA_INVERTED);
    camOverlayStereoPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  {
    RHIShaderDescriptor desc(
      "shaders/ndcQuadXf.vtx.glsl",
      "shaders/camOverlayStereoUndistort.frag.glsl",
      ndcQuadVertexLayout);
    desc.setFlag("CAMERA_INVERTED", (bool) CAMERA_INVERTED);
    camOverlayStereoUndistortPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  {
    RHIShaderDescriptor desc(
    "shaders/ndcClippedQuadXf.vtx.glsl",
    "shaders/camUndistortMask.frag.glsl",
    ndcQuadVertexLayout);
    desc.setFlag("CAMERA_INVERTED", (bool) CAMERA_INVERTED);
    camUndistortMaskPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  {
    RHIShaderDescriptor desc(
      "shaders/ndcQuad.vtx.glsl",
      "shaders/camGreyscale.frag.glsl",
      ndcQuadVertexLayout);
    desc.setFlag("CAMERA_INVERTED", (bool) CAMERA_INVERTED);
    camGreyscalePipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  {
    RHIShaderDescriptor desc(
      "shaders/ndcQuad.vtx.glsl",
      "shaders/camGreyscaleUndistort.frag.glsl",
      ndcQuadVertexLayout);
    desc.setFlag("CAMERA_INVERTED", (bool) CAMERA_INVERTED);
    camGreyscaleUndistortPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  {
    RHIShaderDescriptor desc(
      "shaders/solidQuad.vtx.glsl",
      "shaders/solidQuad.frag.glsl",
      ndcQuadVertexLayout);
    solidQuadPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  viveDistortionPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(RHIShaderDescriptor(
    "shaders/hmdDistortion.vtx.glsl",
    "shaders/viveDistortion.frag.glsl",
    ndcQuadVertexLayout)),
    tristripPipelineDescriptor);


  {
    uint8_t* maskData = new uint8_t[8 * 8];
    memset(maskData, 0xff, 8 * 8);
    disabledMaskTex = rhi()->newTexture2D(8, 8, RHISurfaceDescriptor(kSurfaceFormat_R8));
    rhi()->loadTextureData(disabledMaskTex, kVertexElementTypeUByte1N, maskData);
    delete[] maskData;
  }

  hmdContext = ohmd_ctx_create();
  int num_devices = ohmd_ctx_probe(hmdContext);
  if (num_devices < 0){
    printf("OpenHMD: failed to probe devices: %s\n", ohmd_ctx_get_error(hmdContext));
    return false;
  }

  {
    ohmd_device_settings* hmdSettings = ohmd_device_settings_create(hmdContext);

    hmdDevice = ohmd_list_open_device_s(hmdContext, 0, hmdSettings);
    if (!hmdDevice){
      printf("OpenHMD: failed to open device: %s\n", ohmd_ctx_get_error(hmdContext));
      return false;
    }

    // Not used after ohmd_list_open_device_s returns
    ohmd_device_settings_destroy(hmdSettings);

    // Grab some fixed parameters
    ohmd_device_geti(hmdDevice, OHMD_SCREEN_HORIZONTAL_RESOLUTION, (int*) &hmd_width);
    ohmd_device_geti(hmdDevice, OHMD_SCREEN_VERTICAL_RESOLUTION, (int*) &hmd_height);
    // Eye target dimensions are twice the per-eye screen resolution (which is hmd_width/2 x hmd_height), rounded up to the next 16 pixel block
    eye_width = (hmd_width + 0xf) & ~0xfUL;
    eye_height = ((2 * hmd_height) + 0xf) & ~0xfUL;
    printf("HMD dimensions: %u x %u\n", hmd_width, hmd_height);
  }

  // Set up uniform buffers for HMD distortion passes
  recomputeHMDParameters();

  // Create FBOs for per-eye rendering (pre distortion)
  for (int i = 0; i < 2; ++i) {
    eyeTex[i] = rhi()->newTexture2D(eye_width, eye_height, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    eyeRT[i] = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({ eyeTex[i] }));
  }

  printf("Screen dimensions: %u x %u\n", windowRenderTarget->width(), windowRenderTarget->height());

  if (windowRenderTarget->width() == hmd_width && windowRenderTarget->height() == hmd_height) {
    // Screen physical orientation matches HMD logical orientation
  } else if (windowRenderTarget->width() == hmd_height && windowRenderTarget->height() == hmd_width) {
    // Screen is oriented opposite of HMD logical orientation
    rotate_screen = true;
    printf("Will compensate for screen rotation.\n");
  } else {
    printf("WARNING: Screen and HMD dimensions don't match; check system configuration.\n");
  }

  // RTSP server and NvEnc setup
  {
    nvencSession = new NvEncSession();
    nvencSession->setDimensions(1920, 1080); // should be overwritten by renderSetDebugSurfaceSize
    nvencSession->setInputFormat(NvEncSession::kInputFormatRGBX8);
    nvencSession->setUseGPUFrameSubmission(true);
    //nvencSession->setBitrate(bitrate);
    //nvencSession->setFramerate(fps_n, fps_d);
    nvencSession->setFramerate(30, 1); // TODO derive this from the screen's framerate.
    

    // Begin by setting up our usage environment:
    rtspScheduler = BasicTaskScheduler::createNew();
    rtspEnv = BasicUsageEnvironment::createNew(*rtspScheduler);
    OutPacketBuffer::maxSize = 1048576;

    // Create the RTSP server:
    rtspServer = RTSPServer::createNew(*rtspEnv, 8554);
    if (rtspServer == NULL) {
      printf("Failed to create RTSP server: %s\n", rtspEnv->getResultMsg());
      exit(1);
    }

    char const* descriptionString = "Live555 embedded stream";
    char const* streamName = "0";
    rtspMediaSession = ServerMediaSession::createNew(*rtspEnv, streamName, streamName, descriptionString);
    rtspMediaSession->addSubsession(H264VideoNvEncSessionServerMediaSubsession::createNew(*rtspEnv, nvencSession));
    rtspServer->addServerMediaSession(rtspMediaSession);

    char* urlTmp = rtspServer->rtspURL(rtspMediaSession);
    printf("RTSP server is listening at %s\n", urlTmp);
    printf("Recommended client configuration for low-latency streaming:\n");
    printf("  ffplay -fflags nobuffer -flags low_delay -framedrop %s\n", urlTmp);

    rtspURL = std::string(urlTmp);
    delete[] urlTmp;

    pthread_t server_tid;
    pthread_create(&server_tid, NULL, &rtspServerThreadEntryPoint, NULL);
  }

  return true;
}

void RenderShutdown() {
  // Release OpenGL resources
  eglMakeCurrent( demoState.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT );
  eglDestroySurface( demoState.display, demoState.surface );
  eglDestroyContext( demoState.display, demoState.context );
  eglTerminate( demoState.display );
}

EGLDisplay renderEGLDisplay() { return demoState.display; }
EGLContext renderEGLContext() { return demoState.context; }

void recomputeHMDParameters() {
  float ipd;
  //float sep;

  ohmd_device_getf(hmdDevice, OHMD_EYE_IPD, &ipd);
  //ohmd_device_getf(hmdDevice, OHMD_LENS_HORIZONTAL_SEPARATION, &sep);

  // Setup projection matrices
  // TODO: read/compute this from the Vive config (look at how Monado does it)
  glm::vec4 eyeFovs[2] = {
    /*left, right, top, bottom*/
    glm::vec4(-0.986542, 0.913441, 0.991224, -0.991224),
    glm::vec4(-0.932634, 0.967350, 0.990125, -0.990125)
  };

  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    float projLeft = eyeFovs[eyeIdx][0];
    float projRight = eyeFovs[eyeIdx][1];
#if 0
    float projTop = eyeFovs[eyeIdx][2];
    float projBottom = eyeFovs[eyeIdx][3];
#else
    // flipped
    float projBottom = eyeFovs[eyeIdx][2];
    float projTop = eyeFovs[eyeIdx][3];
#endif

    float idx = 1.0f / (projRight - projLeft);
    float idy = 1.0f / (projBottom - projTop);
    float sx = projRight + projLeft;
    float sy = projBottom + projTop;

    float zNear = 1.0f;

    eyeProjection[eyeIdx] = glm::mat4(
      2.0f*idx,  0.0f,       0.0f,    0.0f,
      0.0f,      2.0f*idy,   0.0f,    0.0f,
      sx*idx,    sy*idy,     0.0f,   -1.0f,
      0.0f,      0.0f,      zNear,    0.0f);
  }

  // Cook the stereo separation transform into the projection matrices
  // TODO correct eye offsets
  eyeView[0] = glm::translate(glm::vec3(ipd *  stereoSeparationScale, 0.0f, 0.0f));
  eyeView[1] = glm::translate(glm::vec3(ipd * -stereoSeparationScale, 0.0f, 0.0f));

  for (size_t i = 0; i < 2; ++i) {
    printf("Eye %zu projection matrix:\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n\n", i,
      eyeProjection[i][0][0], eyeProjection[i][0][1], eyeProjection[i][0][2], eyeProjection[i][0][3],
      eyeProjection[i][1][0], eyeProjection[i][1][1], eyeProjection[i][1][2], eyeProjection[i][1][3],
      eyeProjection[i][2][0], eyeProjection[i][2][1], eyeProjection[i][2][2], eyeProjection[i][2][3],
      eyeProjection[i][3][0], eyeProjection[i][3][1], eyeProjection[i][3][2], eyeProjection[i][3][3]);
  }

  // TODO read vive distortion parameters from JSON config instead of hard-coding them
  // Note that the coeffs[] array is transposed from the storage of in the JSON config
  // JSON stores { distortion_red : { coeffs : [rx, ry, rz, 0] }, distortion : { coeffs : [gx, gy, gz, 0] }, distortion_blue : { coeffs : [bx, by, bz, 0] } }
  // Coeffs array is: {
  // coeffs[0] = (rx, gx, bx, 0)
  // coeffs[1] = (ry, gy, by, 0)
  // coeffs[2] = (rz, gz, bz, 0)
  // }
  {
    ViveDistortionUniformBlock ub;
    ub.coeffs[0] = glm::vec4(-0.187709024978468, -0.2248243919182109, -0.2650347859647872, 0.0);
    ub.coeffs[1] = glm::vec4(-0.08699418167995299, -0.02890679801668017, 0.03408880667124125, 0.0);
    ub.coeffs[2] = glm::vec4(-0.008524150931075117, -0.04008145037518276, -0.07739435170293799, 0.0);

    ub.center = glm::vec4(0.0895289183308623, -0.005774193813369232, 0.0, 0.0); // distortion.center_x, distortion.center_y
    ub.undistort_r2_cutoff = 1.114643216133118;
    ub.aspect_x_over_y = 0.8999999761581421; // // physical_aspect_x_over_y, same for both sides
    ub.grow_for_undistort = 0.6000000238418579;

    viveDistortionParams[0] = rhi()->newUniformBufferWithContents(&ub, sizeof(ViveDistortionUniformBlock));
  }

  {
    ViveDistortionUniformBlock ub;
    ub.coeffs[0] = glm::vec4(-0.1850211958007479, -0.2200407667682694, -0.2690216561778251, 0.0);
    ub.coeffs[1] = glm::vec4(-0.08403208842715496, -0.02952833754861919, 0.05386620639519943, 0.0);
    ub.coeffs[2] = glm::vec4(-0.01036514909834557, -0.04015020276712449, -0.08959133710605897, 0.0);
    ub.center = glm::vec4( -0.08759391576262035, -0.004206675752489539, 0.0, 0.0); // distortion.center_x, distortion.center_y
    ub.undistort_r2_cutoff = 1.087415933609009;
    ub.aspect_x_over_y = 0.8999999761581421; // // physical_aspect_x_over_y, same for both sides
    ub.grow_for_undistort = 0.6000000238418579;

    viveDistortionParams[1] = rhi()->newUniformBufferWithContents(&ub, sizeof(ViveDistortionUniformBlock));
  }
}

void renderHMDFrame() {
  // Switch to output framebuffer
  rhi()->beginRenderPass(windowRenderTarget, kLoadInvalidate);

  // Run distortion passes
  for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {

    if (rotate_screen) {
      if (eyeIndex == 0) {
        rhi()->setViewport(RHIRect::xywh(0, 0, windowRenderTarget->width(), windowRenderTarget->height()/2));
      } else {
        rhi()->setViewport(RHIRect::xywh(0, windowRenderTarget->height()/2, windowRenderTarget->width(), windowRenderTarget->height()/2));
      }
    } else {
      if (eyeIndex == 0) {
        rhi()->setViewport(RHIRect::xywh(0, 0, windowRenderTarget->width()/2, windowRenderTarget->height()));
      } else {
        rhi()->setViewport(RHIRect::xywh(windowRenderTarget->width()/2, 0, windowRenderTarget->width()/2, windowRenderTarget->height()));
      }
    }

    rhi()->bindRenderPipeline(viveDistortionPipeline);
    rhi()->loadUniformBlock(ksViveDistortionUniformBlock, viveDistortionParams[eyeIndex]);
    rhi()->loadTexture(ksImageTex, eyeTex[eyeIndex], linearClampSampler);

    rhi()->drawNDCQuad();
  }

  rhi()->endRenderPass(windowRenderTarget);

  rhi()->swapBuffers(windowRenderTarget);
}


void renderSetDebugSurfaceSize(size_t x, size_t y) {
  nvencSession->setDimensions(x, y);
}

RHISurface::ptr renderAcquireDebugSurface() {
  // RTSP server rendering
  // TODO need to rework the frame handoff so the GPU does the buffer copy
  static uint64_t lastFrameSubmissionTimeNs = 0;
  if (nvencSession->isRunning()) {
    uint64_t now = currentTimeNs();
    if (lastFrameSubmissionTimeNs + rtspRenderIntervalNs <= now) {
      lastFrameSubmissionTimeNs = now;

      return nvencSession->acquireSurface(); // might be NULL anyway if the encoder isn't ready
    }
  }

  return RHISurface::ptr();
}

void renderSubmitDebugSurface(RHISurface::ptr debugSurface) {
  nvencSession->submitSurface(debugSurface);
}

