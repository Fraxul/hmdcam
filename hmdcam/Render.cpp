#include "Render.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/gl/GLCommon.h"
#include "rhi/egl/RHIEGLSurfaceRenderTargetGL.h"

#include "nvgldemo.h"

#include "xrt/xrt_instance.h"
#include "xrt/xrt_device.h"
#include "math/m_api.h"
#include "util/u_distortion_mesh.h"

#include "NvEncSession.h"
#include "liveMedia.hh"
#include "BasicUsageEnvironment.hh"
#include "BufferRingSource.h"
#include "H264VideoNvEncSessionServerMediaSubsession.h"
#include <cuda.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <sys/time.h>

extern bool vive_watchman_enable; // hacky; symbol added in xrt/drivers/vive/vive_device.c to disable watchman thread (since we don't use lighthouse tracking)

RHIRenderTarget::ptr windowRenderTarget;

FxAtomicString ksNDCQuadUniformBlock("NDCQuadUniformBlock");
FxAtomicString ksNDCClippedQuadUniformBlock("NDCClippedQuadUniformBlock");
FxAtomicString ksSolidQuadUniformBlock("SolidQuadUniformBlock");

RHIRenderPipeline::ptr camTexturedQuadPipeline;
RHIRenderPipeline::ptr camOverlayPipeline;
RHIRenderPipeline::ptr camUndistortMaskPipeline;
RHIRenderPipeline::ptr camUndistortOverlayPipeline;
RHIRenderPipeline::ptr solidQuadPipeline;

RHISurface::ptr disabledMaskTex;

RHIRenderPipeline::ptr mesh1chDistortionPipeline;
RHIRenderPipeline::ptr mesh3chDistortionPipeline;

FxAtomicString ksOverlayTex("overlayTex");
FxAtomicString ksMaskTex("maskTex");

// per-eye render targets (pre distortion)
RHISurface::ptr eyeTex[2];
RHIRenderTarget::ptr eyeRT[2];

// distortion parameter buffers
RHIBuffer::ptr meshDistortionVertexBuffer, meshDistortionIndexBuffer;

// HMD info/state
struct xrt_instance* xrtInstance = NULL;
struct xrt_device* xrtHMDevice = NULL;

unsigned int hmd_width, hmd_height;
unsigned int eye_width, eye_height;
glm::mat4 eyeProjection[2];
glm::mat4 eyeView[2];

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

  camTexturedQuadPipeline = rhi()->compileRenderPipeline("shaders/ndcQuadXf_vFlip.vtx.glsl", "shaders/camTexturedQuad.frag.glsl", ndcQuadVertexLayout, kPrimitiveTopologyTriangleStrip);
  camOverlayPipeline = rhi()->compileRenderPipeline("shaders/ndcQuadXf_vFlip.vtx.glsl", "shaders/camOverlay.frag.glsl", ndcQuadVertexLayout, kPrimitiveTopologyTriangleStrip);
  camUndistortMaskPipeline = rhi()->compileRenderPipeline("shaders/ndcClippedQuadXf_vFlip.vtx.glsl", "shaders/camUndistortMask.frag.glsl", ndcQuadVertexLayout, kPrimitiveTopologyTriangleStrip);
  camUndistortOverlayPipeline = rhi()->compileRenderPipeline("shaders/ndcQuadXf_vFlip.vtx.glsl", "shaders/camUndistortOverlay.frag.glsl", ndcQuadVertexLayout, kPrimitiveTopologyTriangleStrip);
  solidQuadPipeline = rhi()->compileRenderPipeline("shaders/solidQuad.vtx.glsl", "shaders/solidQuad.frag.glsl", ndcQuadVertexLayout, kPrimitiveTopologyTriangleStrip);

  {
    RHIVertexLayout vtx;
    vtx.elements.push_back(RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "position_Ruv", 0, 16));
    RHIShaderDescriptor desc(
          "shaders/meshDistortion.vtx.glsl",
          "shaders/meshDistortion.frag.glsl",
          vtx);
    desc.setFlag("CHROMA_CORRECTION", false);

    mesh1chDistortionPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  {
    RHIVertexLayout vtx;
    vtx.elements.push_back(RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "position_Ruv", 0,  32));
    vtx.elements.push_back(RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "Guv_Buv",      16, 32));

    RHIShaderDescriptor desc(
          "shaders/meshDistortion.vtx.glsl",
          "shaders/meshDistortion.frag.glsl",
          vtx);
    desc.setFlag("CHROMA_CORRECTION", true);
    mesh3chDistortionPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  {
    uint8_t* maskData = new uint8_t[8 * 8];
    memset(maskData, 0xff, 8 * 8);
    disabledMaskTex = rhi()->newTexture2D(8, 8, RHISurfaceDescriptor(kSurfaceFormat_R8));
    rhi()->loadTextureData(disabledMaskTex, kVertexElementTypeUByte1N, maskData);
    delete[] maskData;
  }

  // Monado setup
  {
    vive_watchman_enable = false; // Skip Watchman initialization, we don't (can't) use lighthouse tracking here.

    int ret;
    const size_t NUM_XDEVS = 32;
    struct xrt_device *xdevs[NUM_XDEVS];
    memset(xdevs, 0, sizeof(struct xrt_device*) * NUM_XDEVS);

    ret = xrt_instance_create(NULL, &xrtInstance);
    if (ret != 0) {
      printf("xrt_instance_create() failed: %d\n", ret);
      return false;
    }

    ret = xrt_instance_select(xrtInstance, xdevs, NUM_XDEVS);
    if (ret != 0) {
      printf("xrt_instance_select() failed: %d\n", ret);
      return false;
    }

    // Select the first HMD and destroy the rest of the devices (if any)
    for (size_t i = 0; i < NUM_XDEVS; i++) {
      if (xdevs[i] == NULL) {
        continue;
      }

      if (xrtHMDevice == NULL && xdevs[i]->device_type == XRT_DEVICE_TYPE_HMD) {
        printf("Selected HMD device: %s\n", xdevs[i]->str);
        xrtHMDevice = xdevs[i];
      } else {
        printf("\tDestroying unused device %s\n", xdevs[i]->str);
        xrt_device_destroy(&xdevs[i]);
      }
    }

    struct xrt_hmd_parts* hmd = xrtHMDevice->hmd; 
    assert(hmd);

    // Dump HMD info
    printf("HMD screen: %d x %d, %lu ns nominal frame interval (%.3f FPS)\n", hmd->screens[0].w_pixels, hmd->screens[0].h_pixels, hmd->screens[0].nominal_frame_interval_ns, 1000000000.0 / static_cast<double>(hmd->screens[0].nominal_frame_interval_ns));
    printf("Viewports:\n"); 
    for (int viewportIdx = 0; viewportIdx < 2; ++viewportIdx) {
      printf("[%d] %u x %u pixels @ %u, %u\n", viewportIdx, hmd->views[viewportIdx].viewport.w_pixels, hmd->views[viewportIdx].viewport.h_pixels, hmd->views[viewportIdx].viewport.x_pixels, hmd->views[viewportIdx].viewport.y_pixels);
    }

    printf("Distortion models: %s%s%s\n",
      hmd->distortion.models & XRT_DISTORTION_MODEL_NONE ? "None " : "",
      hmd->distortion.models & XRT_DISTORTION_MODEL_MESHUV ? "MeshUV " : "",
      hmd->distortion.models & XRT_DISTORTION_MODEL_COMPUTE ? "Compute " : "");

    if (!(hmd->distortion.models & XRT_DISTORTION_MODEL_MESHUV)) {
      if (!((hmd->distortion.models & XRT_DISTORTION_MODEL_NONE) || (hmd->distortion.models & XRT_DISTORTION_MODEL_COMPUTE))) {
        printf("HMD does not report any usable distortion models (MeshUV, Compute, or None)\n");
        return false;
      }
      printf("Generating HMD MeshUV distortion from Compute function\n");
      u_distortion_mesh_fill_in_compute(xrtHMDevice);
    }

    printf("Distortion mesh data:\n");
    printf("vertices=%zu stride=%zu num_uv_channels=%zu num_indices={%zu, %zu} offset_indices={%zu, %zu} total_num_indices=%zu\n",
      hmd->distortion.mesh.num_vertices, hmd->distortion.mesh.stride, hmd->distortion.mesh.num_uv_channels,
      hmd->distortion.mesh.num_indices[0], hmd->distortion.mesh.num_indices[1],
      hmd->distortion.mesh.offset_indices[0], hmd->distortion.mesh.offset_indices[1],
      hmd->distortion.mesh.total_num_indices);

    // Upload vertex and index buffers for distortion
    meshDistortionVertexBuffer = rhi()->newBufferWithContents(hmd->distortion.mesh.vertices, hmd->distortion.mesh.num_vertices * hmd->distortion.mesh.stride);
    meshDistortionIndexBuffer = rhi()->newBufferWithContents(hmd->distortion.mesh.indices, hmd->distortion.mesh.total_num_indices * sizeof(uint32_t));

    // Setup global state
    hmd_width = hmd->screens[0].w_pixels;
    hmd_height = hmd->screens[0].h_pixels;

    // Eye target dimensions are twice the per-eye viewport resolution, rounded up to the next 16 pixel block
    eye_width = ((hmd->views[0].viewport.w_pixels * 2) + 0xf) & ~0xfUL;
    eye_height = ((hmd->views[0].viewport.h_pixels * 2) + 0xf) & ~0xfUL;
    printf("Eye target dimensions: %u x %u\n", eye_width, eye_height);

  } // Monado setup

  // Set up uniform buffers for HMD distortion passes
  recomputeHMDParameters();

  // Create FBOs for per-eye rendering (pre distortion)
  for (int i = 0; i < 2; ++i) {
    eyeTex[i] = rhi()->newTexture2D(eye_width, eye_height, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    eyeRT[i] = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({ eyeTex[i] }));
  }

  printf("Screen dimensions: %u x %u\n", windowRenderTarget->width(), windowRenderTarget->height());

  if (!(windowRenderTarget->width() == hmd_width && windowRenderTarget->height() == hmd_height)) {
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

  if (xrtHMDevice)
    xrt_device_destroy(&xrtHMDevice);

  if (xrtInstance)
    xrt_instance_destroy(&xrtInstance);
}

EGLDisplay renderEGLDisplay() { return demoState.display; }
EGLContext renderEGLContext() { return demoState.context; }

void recomputeHMDParameters() {
  float zNear = 0.5f;

  // from renderer_get_view_projection (compositor/main/comp_renderer.c)
  struct xrt_vec3 eye_relation = {
      0.063000f, /* TODO: get actual ipd_meters */
      0.0f,
      0.0f,
  };

  for (uint32_t eyeIdx = 0; eyeIdx < 2; eyeIdx++) {
    struct xrt_fov* fov = &xrtHMDevice->hmd->views[eyeIdx].fov;

    // from comp_layer_renderer_set_fov
    const float tan_left = tanf(fov->angle_left);
    const float tan_right = tanf(fov->angle_right);

    const float tan_down = tanf(fov->angle_down);
    const float tan_up = tanf(fov->angle_up);

    const float tan_width = tan_right - tan_left;
    const float tan_height = tan_up - tan_down;

    const float a11 = 2.0f / tan_width;
    const float a22 = 2.0f / tan_height;

    const float a31 = (tan_right + tan_left) / tan_width;
    const float a32 = (tan_up + tan_down) / tan_height;
    
    /*
    self->mat_projection[eye] = (struct xrt_matrix_4x4) {
      .v = {
        a11, 0, 0, 0,
        0, a22, 0, 0,
        a31, a32, a33, -1,
        0, 0, a43, 0,
      }
    };*/

    // Right-handed infinite-Z far plane
    eyeProjection[eyeIdx] = glm::mat4(
       a11,  0.0f,  0.0f,   0.0f,
      0.0f,   a22,  0.0f,   0.0f,
       a31,   a32,  0.0f,  -1.0f,
      0.0f,  0.0f,  zNear,  0.0f);

    struct xrt_pose eye_pose;
    xrt_device_get_view_pose(xrtHMDevice, &eye_relation, eyeIdx, &eye_pose);

    xrt_matrix_4x4 eye_view;
    math_matrix_4x4_view_from_pose(&eye_pose, &eye_view);

    const float* v = eye_view.v;
    eyeView[eyeIdx] = glm::mat4(
      v[ 0], v[ 1], v[ 2], v[ 3],
      v[ 4], v[ 5], v[ 6], v[ 7],
      v[ 8], v[ 9], v[10], v[11],
      v[12], v[13], v[14], v[15]);
  }

  for (size_t i = 0; i < 2; ++i) {
    printf("Eye %zu projection matrix:\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n\n", i,
      eyeProjection[i][0][0], eyeProjection[i][0][1], eyeProjection[i][0][2], eyeProjection[i][0][3],
      eyeProjection[i][1][0], eyeProjection[i][1][1], eyeProjection[i][1][2], eyeProjection[i][1][3],
      eyeProjection[i][2][0], eyeProjection[i][2][1], eyeProjection[i][2][2], eyeProjection[i][2][3],
      eyeProjection[i][3][0], eyeProjection[i][3][1], eyeProjection[i][3][2], eyeProjection[i][3][3]);
  }
}

void renderHMDFrame() {
  // Switch to output framebuffer
  rhi()->beginRenderPass(windowRenderTarget, kLoadInvalidate);

  if (xrtHMDevice->hmd->distortion.mesh.num_uv_channels == 1) {
    rhi()->bindRenderPipeline(mesh1chDistortionPipeline);
  } else {
    rhi()->bindRenderPipeline(mesh3chDistortionPipeline);
  }

  rhi()->bindStreamBuffer(0, meshDistortionVertexBuffer);

  // Run distortion passes
  for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {

    rhi()->setViewport(RHIRect::xywh(
      xrtHMDevice->hmd->views[eyeIndex].viewport.x_pixels,
      xrtHMDevice->hmd->views[eyeIndex].viewport.y_pixels,
      xrtHMDevice->hmd->views[eyeIndex].viewport.w_pixels,
      xrtHMDevice->hmd->views[eyeIndex].viewport.h_pixels));

    rhi()->loadTexture(ksImageTex, eyeTex[eyeIndex], linearClampSampler);

    rhi()->drawIndexedPrimitives(meshDistortionIndexBuffer, kIndexBufferTypeUInt32, xrtHMDevice->hmd->distortion.mesh.num_indices[eyeIndex], xrtHMDevice->hmd->distortion.mesh.offset_indices[eyeIndex]);
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

const std::string& renderDebugURL() {
  return rtspURL;
}

