#include "Render.h"
#include "RenderBackend.h"
#include "RenderBackendWayland.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/gl/GLCommon.h"

#include "xrt/xrt_instance.h"
#include "xrt/xrt_device.h"
#include "math/m_api.h"
#include "util/u_distortion_mesh.h"
#include "util/u_device.h"

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

// combined eye render target (pre distortion)
RHISurface::ptr eyeTex;
RHISurface::ptr eyeDepthRenderbuffer;
RHIRenderTarget::ptr eyeRT;
RHIRect eyeViewports[2]; // viewports in the eye RT
RHIRect eyePostDistortionViewports[2]; // viewports on the HMD window surface

// distortion parameter buffers
RHIBuffer::ptr meshDistortionVertexBuffer, meshDistortionIndexBuffer;
struct MeshDistortionUniformBlock {
  glm::vec2 uvOffset;
  glm::vec2 uvScale;
};
static FxAtomicString ksMeshDistortionUniformBlock("MeshDistortionUniformBlock");

// HMD info/state
struct xrt_instance* xrtInstance = NULL;
struct xrt_device* xrtHMDevice = NULL;

bool isDummyHMD = false;
unsigned int hmd_width, hmd_height;
unsigned int eye_width, eye_height;
glm::mat4 eyeProjection[2];
glm::mat4 eyeView[2];

RenderBackend* renderBackend = NULL;

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
  pthread_setname_np(pthread_self(), "RTSP-Server");

  printf("Starting RTSP server event loop\n");
  EGLint ctxAttrs[] = {
    EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE
  };

  EGLContext eglCtx = eglCreateContext(renderBackend->eglDisplay(), renderBackend->eglConfig(), renderBackend->eglContext(), ctxAttrs);
  if (!eglCtx) {
    die("rtspServerThreadEntryPoint: unable to create EGL share context\n");
  }

  bool res = eglMakeCurrent(renderBackend->eglDisplay(), EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx);
  if (!res) {
    die("rtspServerThreadEntryPoint: eglMakeCurrent() failed\n");
  }

  cuCtxSetCurrent(cudaContext);

  rtspEnv->taskScheduler().doEventLoop();

  eglDestroyContext(renderBackend->eglDisplay(), eglCtx);

  return NULL;
}

bool RenderInit(ERenderBackend backendType) {
  // Monado setup -- this needs to occur before EGL initialization because we might need to send a command to turn on the HMD display.
  struct xrt_hmd_parts* hmd = NULL;
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
        if (strstr(xdevs[i]->str, "Dummy HMD")) {
          isDummyHMD = true;
        }
        xrtHMDevice = xdevs[i];
      } else {
        printf("\tDestroying unused device %s\n", xdevs[i]->str);
        xrt_device_destroy(&xdevs[i]);
      }
    }

    hmd = xrtHMDevice->hmd;
    assert(hmd);

    // Dump HMD info
    printf("HMD screen: %d x %d, %lu ns nominal frame interval (%.3f FPS)\n", hmd->screens[0].w_pixels, hmd->screens[0].h_pixels, hmd->screens[0].nominal_frame_interval_ns, 1000000000.0 / static_cast<double>(hmd->screens[0].nominal_frame_interval_ns));
    printf("Viewports:\n");
    for (int viewportIdx = 0; viewportIdx < 2; ++viewportIdx) {
      printf("[%d] %u x %u pixels @ %u, %u\n", viewportIdx, hmd->views[viewportIdx].viewport.w_pixels, hmd->views[viewportIdx].viewport.h_pixels, hmd->views[viewportIdx].viewport.x_pixels, hmd->views[viewportIdx].viewport.y_pixels);
    }

    // Setup global state
    hmd_width = hmd->screens[0].w_pixels;
    hmd_height = hmd->screens[0].h_pixels;

    // Eye target dimensions are twice the per-eye viewport resolution, rounded up to the next 16 pixel block
    eye_width = ((hmd->views[0].viewport.w_pixels * 2) + 0xf) & ~0xfUL;
    eye_height = ((hmd->views[0].viewport.h_pixels * 2) + 0xf) & ~0xfUL;
    printf("Eye target dimensions: %u x %u\n", eye_width, eye_height);

  }

  // EGL/DRM setup
  renderBackend = RenderBackend::create(backendType);
  renderBackend->init();
  windowRenderTarget = renderBackend->windowRenderTarget();

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


  // Set up distortion models
  {
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
    printf("vertices=%u stride=%u uv_channels_count=%u index_counts={%u, %u} index_offsets={%u, %u} index_count_total=%u\n",
      hmd->distortion.mesh.vertex_count, hmd->distortion.mesh.stride, hmd->distortion.mesh.uv_channels_count,
      hmd->distortion.mesh.index_counts[0], hmd->distortion.mesh.index_counts[1],
      hmd->distortion.mesh.index_offsets[0], hmd->distortion.mesh.index_offsets[1],
      hmd->distortion.mesh.index_count_total);

    // Upload vertex and index buffers for distortion
    meshDistortionVertexBuffer = rhi()->newBufferWithContents(hmd->distortion.mesh.vertices, hmd->distortion.mesh.vertex_count * hmd->distortion.mesh.stride);
    meshDistortionIndexBuffer = rhi()->newBufferWithContents(hmd->distortion.mesh.indices, hmd->distortion.mesh.index_count_total * sizeof(uint32_t));

    // Compute post-distortion viewports
    for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
      eyePostDistortionViewports[eyeIndex] = RHIRect::xywh(
          hmd->views[eyeIndex].viewport.x_pixels,
          hmd->views[eyeIndex].viewport.y_pixels,
          hmd->views[eyeIndex].viewport.w_pixels,
          hmd->views[eyeIndex].viewport.h_pixels);
    }
  } // Monado distortion setup

  // Set up uniform buffers for HMD distortion passes
  recomputeHMDParameters();

  printf("Screen dimensions: %u x %u\n", windowRenderTarget->width(), windowRenderTarget->height());
  if (isDummyHMD) {
    // Resize the dummy HMD eye RTs to match the attached display.
    hmd_width = windowRenderTarget->width();
    hmd_height = windowRenderTarget->height();
    // Use 1:1 eye targets since we have no distortion to compensate for
    eye_width = hmd_width / 2;
    eye_height = hmd_height;
    // Reset the viewports
    eyePostDistortionViewports[0] = RHIRect::xywh(        0, 0, eye_width, eye_height);
    eyePostDistortionViewports[1] = RHIRect::xywh(eye_width, 0, eye_width, eye_height);
  }

  if (!(windowRenderTarget->width() == hmd_width && windowRenderTarget->height() == hmd_height)) {
    printf("WARNING: Screen and HMD dimensions don't match; check system configuration.\n");
  }

  // Create FBOs and viewports for eye rendering (pre distortion)
  eyeTex = rhi()->newTexture2D(eye_width * 2, eye_height, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
  eyeDepthRenderbuffer = rhi()->newRenderbuffer2D(eye_width * 2, eye_height, RHISurfaceDescriptor(kSurfaceFormat_Depth16));
  eyeRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({ eyeTex }, eyeDepthRenderbuffer));
  eyeViewports[0] = RHIRect::xywh(0, 0, eye_width, eye_height);
  eyeViewports[1] = RHIRect::xywh(eye_width, 0, eye_width, eye_height);

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
  delete renderBackend;
  renderBackend = NULL;

  if (xrtHMDevice)
    xrt_device_destroy(&xrtHMDevice);

  if (xrtInstance)
    xrt_instance_destroy(&xrtInstance);
}

void recomputeHMDParameters() {
  float zNear = 0.0f;

  // from renderer_get_view_projection (compositor/main/comp_renderer.c)
  struct xrt_vec3 eye_relation = {
      0.063000f, /* TODO: get actual ipd_meters */
      0.0f,
      0.0f,
  };

  for (uint32_t eyeIdx = 0; eyeIdx < 2; eyeIdx++) {
    struct xrt_fov* fov = &xrtHMDevice->hmd->distortion.fov[eyeIdx];

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
    u_device_get_view_pose(&eye_relation, eyeIdx, &eye_pose);

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

  if (xrtHMDevice->hmd->distortion.mesh.uv_channels_count == 1) {
    rhi()->bindRenderPipeline(mesh1chDistortionPipeline);
  } else {
    rhi()->bindRenderPipeline(mesh3chDistortionPipeline);
  }

  rhi()->bindStreamBuffer(0, meshDistortionVertexBuffer);
  rhi()->loadTexture(ksImageTex, eyeTex, linearClampSampler);

  // Run distortion passes
  for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {

    rhi()->setViewport(eyePostDistortionViewports[eyeIndex]);

    MeshDistortionUniformBlock ub;
    ub.uvOffset = glm::vec2(eyeIndex == 0 ? 0.0f : 0.5f, 0.0f);
    ub.uvScale = glm::vec2(0.5f, 1.0f);
    rhi()->loadUniformBlockImmediate(ksMeshDistortionUniformBlock, &ub, sizeof(ub));

    rhi()->drawIndexedPrimitives(meshDistortionIndexBuffer, kIndexBufferTypeUInt32, xrtHMDevice->hmd->distortion.mesh.index_counts[eyeIndex], xrtHMDevice->hmd->distortion.mesh.index_offsets[eyeIndex]);
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

