#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/gl/RHIEGLSurfaceRenderTargetGL.h"

#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <GLES3/gl3ext.h>
#include "EGL/egl.h"
#include "EGL/eglext.h"
#include "nvgldemo.h"

#include "ArgusCamera.h"
#include "GLUtils.h"

#include "openhmd/openhmd.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>


//#define SINGLE_CAMERA
#define SWAP_CAMERA_EYES


#ifdef SWAP_CAMERA_EYES
  #define LEFT_CAMERA_INDEX 1
  #define RIGHT_CAMERA_INDEX 0
#else
  #define LEFT_CAMERA_INDEX 0
  #define RIGHT_CAMERA_INDEX 1
#endif

std::array<double, 9> s_cameraMatrix = { 8.1393520905199455e+02, 0., 6.4611705518491897e+02, 0., 8.1393520905199455e+02, 3.7468428117333934e+02, 0., 0., 1. };
std::array<double, 5> s_distortionCoeffs = { -3.7085816639967079e-01, 1.9997393684065998e-01, -2.3017909433031760e-04, 2.7313395926290304e-06, -6.9489964467138884e-02 };


RHIRenderTarget::ptr windowRenderTarget;

struct NDCQuadUniformBlock {
  glm::mat4 modelViewProjection;
};
FxAtomicString ksNDCQuadUniformBlock("NDCQuadUniformBlock");
RHIRenderPipeline::ptr camTexturedQuadPipeline;
RHIRenderPipeline::ptr camInvDistortionPipeline;

struct HMDDistortionUniformBlock {
  glm::vec4 hmdWarpParam;
  glm::vec4 aberr; // actually vec3, padded
  glm::vec2 lensCenter;
  glm::vec2 viewportScale;
  float warpScale;
  float pad2, pad3, pad4;
};
FxAtomicString ksHMDDistortionUniformBlock("HMDDistortionUniformBlock");
RHIRenderPipeline::ptr hmdDistortionPipeline;

FxAtomicString ksImageTex("imageTex");
FxAtomicString ksDistortionMap("distortionMap");

// per-eye render targets (pre distortion)
RHISurface::ptr eyeTex[2];
RHIRenderTarget::ptr eyeRT[2];
// per-eye distortion parameter buffers
RHIBuffer::ptr hmdDistortionParams[2];


typedef struct
{
  // HMD info/state
  ohmd_context* hmdContext;
  ohmd_device* hmdDevice;

  int hmd_width, hmd_height;
  int eye_width, eye_height;
  glm::mat4 eyeProjection[2];

  bool rotate_screen;

  int verbose;

} CUBE_STATE_T;

NvGlDemoOptions demoOptions;
static CUBE_STATE_T state;

static void init_ogl() {
  memset(&demoOptions, 0, sizeof(demoOptions));

  demoOptions.displayAlpha = 1.0;
  demoOptions.nFifo = 1;

  // Use the current mode and the entire screen
  demoOptions.useCurrentMode = 1;
  demoOptions.windowSize[0] = 0;
  demoOptions.windowSize[1] = 0;

  NvGlDemoInitializeEGL(0, 0);
  printf("%s\n", glGetString(GL_RENDERER));
  printf("%s\n", glGetString(GL_VERSION));
  printf("%s\n", glGetString(GL_EXTENSIONS));

  initRHIGL();

  RHIEGLSurfaceRenderTargetGL::ptr wrt(new RHIEGLSurfaceRenderTargetGL(demoState.display, demoState.surface));
  wrt->platformSetUpdatedWindowDimensions(demoState.width, demoState.height);
  windowRenderTarget = wrt;

  // Set up shared resources

  camTexturedQuadPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(RHIShaderDescriptor(
    "shaders/ndcQuad.vtx.glsl",
    "shaders/camTexturedQuad.frag.glsl",
    ndcQuadVertexLayout)),
    tristripPipelineDescriptor);

  camInvDistortionPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(RHIShaderDescriptor(
    "shaders/ndcQuad.vtx.glsl",
    "shaders/camInvDistortion.frag.glsl",
    ndcQuadVertexLayout)),
    tristripPipelineDescriptor);

  hmdDistortionPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(RHIShaderDescriptor(
    "shaders/hmdDistortion.vtx.glsl",
    "shaders/hmdDistortion.frag.glsl",
    ndcQuadVertexLayout)),
    tristripPipelineDescriptor);
}

static void update_fps() {
   static int frame_count = 0;
   static long long time_start = 0;
   long long time_now;
   struct timeval te;
   float fps;

   frame_count++;

   gettimeofday(&te, NULL);
   time_now = te.tv_sec * 1000LL + te.tv_usec / 1000;

   if (time_start == 0)
   {
      time_start = time_now;
   }
   else if (time_now - time_start > 5000)
   {
      fps = (float) frame_count / ((time_now - time_start) / 1000.0);
      frame_count = 0;
      time_start = time_now;
      fprintf(stderr, "%3.2f FPS\n", fps);
   }
}

static bool want_quit = false;
static void signal_handler(int) {
  want_quit = true;

  // Restore signal handlers so the program is still interruptable if clean shutdown gets stuck
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);
}

void recomputeHMDParameters() {
  float ipd;
  glm::vec2 viewport_scale;
  glm::vec4 distortion_coeffs;
  glm::vec3 aberr_scale;
  float sep;
  glm::vec2 left_lens_center;
  glm::vec2 right_lens_center;
  float warp_scale;
  float warp_adj;

  ohmd_device_getf(state.hmdDevice, OHMD_EYE_IPD, &ipd);
  //viewport is half the screen
  ohmd_device_getf(state.hmdDevice, OHMD_SCREEN_HORIZONTAL_SIZE, &(viewport_scale[0]));
  viewport_scale[0] /= 2.0f;
  ohmd_device_getf(state.hmdDevice, OHMD_SCREEN_VERTICAL_SIZE, &(viewport_scale[1]));
  //distortion coefficients
  ohmd_device_getf(state.hmdDevice, OHMD_UNIVERSAL_DISTORTION_K, &(distortion_coeffs[0]));
  ohmd_device_getf(state.hmdDevice, OHMD_UNIVERSAL_ABERRATION_K, &(aberr_scale[0]));
  //calculate lens centers (assuming the eye separation is the distance between the lens centers)
  ohmd_device_getf(state.hmdDevice, OHMD_LENS_HORIZONTAL_SEPARATION, &sep);
  ohmd_device_getf(state.hmdDevice, OHMD_LENS_VERTICAL_POSITION, &(left_lens_center[1]));
  ohmd_device_getf(state.hmdDevice, OHMD_LENS_VERTICAL_POSITION, &(right_lens_center[1]));
  left_lens_center[0] = viewport_scale[0] - sep/2.0f;
  right_lens_center[0] = sep/2.0f;
  //assume calibration was for lens view to which ever edge of screen is further away from lens center
  warp_scale = (left_lens_center[0] > right_lens_center[0]) ? left_lens_center[0] : right_lens_center[0];
  warp_adj = 1.0f;

  // Setup projection matrices
  ohmd_device_getf(state.hmdDevice, OHMD_LEFT_EYE_GL_PROJECTION_MATRIX, &(state.eyeProjection[0][0][0]));
  ohmd_device_getf(state.hmdDevice, OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX, &(state.eyeProjection[1][0][0]));
  // Cook the stereo separation transform into the projection matrices
  // TODO stereo separation scale
  state.eyeProjection[0] = state.eyeProjection[0] * glm::translate(glm::vec3(ipd *  10.0f, 0.0f, 0.0f));
  state.eyeProjection[1] = state.eyeProjection[1] * glm::translate(glm::vec3(ipd * -10.0f, 0.0f, 0.0f));

  for (size_t eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
    HMDDistortionUniformBlock ub;
    ub.hmdWarpParam = distortion_coeffs;
    ub.aberr = glm::vec4(aberr_scale, 0.0f);
    ub.lensCenter = (eyeIndex == 0 ? left_lens_center : right_lens_center);
    ub.viewportScale = viewport_scale;
    ub.warpScale = warp_scale * warp_adj;

    hmdDistortionParams[eyeIndex] = rhi()->newUniformBufferWithContents(&ub, sizeof(HMDDistortionUniformBlock));
  }
}

int main(int argc, char* argv[]) {

  memset(&state, 0, sizeof(state));

  init_ogl();



  state.hmdContext = ohmd_ctx_create();
  int num_devices = ohmd_ctx_probe(state.hmdContext);
  if (num_devices < 0){
    printf("OpenHMD: failed to probe devices: %s\n", ohmd_ctx_get_error(state.hmdContext));
    return 1;
  }

  {
    ohmd_device_settings* hmdSettings = ohmd_device_settings_create(state.hmdContext);

    state.hmdDevice = ohmd_list_open_device_s(state.hmdContext, 0, hmdSettings);
    if (!state.hmdDevice){
      printf("OpenHMD: failed to open device: %s\n", ohmd_ctx_get_error(state.hmdContext));
      return 1;
    }

    // Not used after ohmd_list_open_device_s returns
    ohmd_device_settings_destroy(hmdSettings);

    // Grab some fixed parameters
    ohmd_device_geti(state.hmdDevice, OHMD_SCREEN_HORIZONTAL_RESOLUTION, &state.hmd_width);
    ohmd_device_geti(state.hmdDevice, OHMD_SCREEN_VERTICAL_RESOLUTION, &state.hmd_height);
    state.eye_width = state.hmd_width / 2;
    state.eye_height = state.hmd_height;
    printf("HMD dimensions: %u x %u\n", state.hmd_width, state.hmd_height);
  }

  // Set up uniform buffers for HMD distortion passes
  recomputeHMDParameters();

  // Create FBOs for per-eye rendering (pre distortion)
  for (int i = 0; i < 2; ++i) {
    eyeTex[i] = rhi()->newTexture2D(state.eye_width, state.eye_height, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    eyeRT[i] = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({ eyeTex[i] }));
  }

  printf("Screen dimensions: %u x %u\n", windowRenderTarget->width(), windowRenderTarget->height());

  if (windowRenderTarget->width() == state.hmd_width && windowRenderTarget->height() == state.hmd_height) {
    // Screen physical orientation matches HMD logical orientation
  } else if (windowRenderTarget->width() == state.hmd_height && windowRenderTarget->height() == state.hmd_width) {
    // Screen is oriented opposite of HMD logical orientation
    state.rotate_screen = true;
    printf("Will compensate for screen rotation.\n");
  } else {
    printf("WARNING: Screen and HMD dimensions don't match; check system configuration.\n");
  }


  ArgusCamera* camera[2];

  // Left
  camera[0] = new ArgusCamera(demoState.display, demoState.context, LEFT_CAMERA_INDEX, 1280, 720);

#ifdef SINGLE_CAMERA
  // Left and right share the same camera
  camera[1] = camera[0];
#else
  // Right
  camera[1] = new ArgusCamera(demoState.display, demoState.context, RIGHT_CAMERA_INDEX, 1280, 720);
#endif

  RHISurface::ptr distortionMap;

  {

    cv::Mat cameraMatrix(cv::Size(3, 3), CV_64F, &(s_cameraMatrix[0]));
    cv::Mat distCoeff(s_distortionCoeffs);
    cv::Size imageSize = cv::Size(1280, 720);
    float alpha = 0.25; // scaling factor. 0 = no invalid pixels in output (no black borders), 1 = use all input pixels
#if 1
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeff, imageSize, alpha, cv::Size(), NULL, /*centerPrincipalPoint=*/true);
#else
    cv::Mat newCameraMatrix = cv::getDefaultNewCameraMatrix(cameraMatrix, imageSize, true);
#endif
    cv::Mat map1, map2;
    cv::initUndistortRectifyMap(cameraMatrix, distCoeff, cv::noArray(), newCameraMatrix, imageSize, CV_32F, map1, map2);
    // map1 and map2 should contain absolute x and y coords for sampling the input image, in pixel scale (map1 is 0-1280, map2 is 0-720, etc)

    // Combine the maps into a buffer we can upload to opengl. Remap the absolute pixel coordinates to UV (0...1) range to save work in the pixel shader.
    float* distortionMapTmp = new float[imageSize.width * imageSize.height * 2];
    for (int y = 0; y < imageSize.height; ++y) {
      for (int x = 0; x < imageSize.width; ++x) {
        // .at(row, col) -- Y rows, X columns.
        distortionMapTmp[(((y * imageSize.width) + x) * 2) + 0] = map1.at<float>(y, x) / static_cast<float>(imageSize.width);
        distortionMapTmp[(((y * imageSize.width) + x) * 2) + 1] = map2.at<float>(y, x) / static_cast<float>(imageSize.height);
      }
    }

    distortionMap = rhi()->newTexture2D(imageSize.width, imageSize.height, RHISurfaceDescriptor(kSurfaceFormat_RG32f));
    rhi()->loadTextureData(distortionMap, kVertexElementTypeFloat2, distortionMapTmp);

    delete[] distortionMapTmp;
  }

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);

  while (!want_quit)
  {
    camera[0]->readFrame();
#ifndef SINGLE_CAMERA
    camera[1]->readFrame();
#endif


    for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
      ArgusCamera* activeCamera = camera[eyeIndex];

      rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
      rhi()->beginRenderPass(eyeRT[eyeIndex], kLoadClear);

#if 1
      rhi()->bindRenderPipeline(camInvDistortionPipeline);
      rhi()->loadTexture(ksImageTex, activeCamera->rgbTexture());
      rhi()->loadTexture(ksDistortionMap, distortionMap);


      // coordsys right now: -X = left, -Z = into screen
      // (camera is at the origin)
      const glm::vec3 tx = glm::vec3(0.0f, 0.0f, -7.0f);
      const float scaleFactor = 5.0f;
      glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(-scaleFactor * (static_cast<float>(activeCamera->streamWidth()) / static_cast<float>(activeCamera->streamHeight())), scaleFactor, 1.0f)); // TODO
      glm::mat4 mvp = state.eyeProjection[eyeIndex] * model;

      NDCQuadUniformBlock ub;
      ub.modelViewProjection = mvp;
      rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(NDCQuadUniformBlock));

      rhi()->drawNDCQuad();
#endif

      rhi()->endRenderPass(eyeRT[eyeIndex]);
    }


    // Switch to output framebuffer
    rhi()->setClearColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    rhi()->beginRenderPass(windowRenderTarget, kLoadClear);

    // XXX TODO
#if 0
    rhi()->blitTex(eyeTex[0], 0);
#else

    // Run distortion passes
    for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {

      if (state.rotate_screen) {
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

      rhi()->bindRenderPipeline(hmdDistortionPipeline);
      rhi()->loadUniformBlock(ksHMDDistortionUniformBlock, hmdDistortionParams[eyeIndex]);
      rhi()->loadTexture(ksImageTex, eyeTex[eyeIndex]);

      rhi()->drawNDCQuad();
    }
#endif

    rhi()->endRenderPass(windowRenderTarget);

    rhi()->swapBuffers(windowRenderTarget);

    update_fps();
  }

  // Restore signal handlers
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);

  // clear screen
  rhi()->beginRenderPass(windowRenderTarget, kLoadClear);
  rhi()->endRenderPass(windowRenderTarget);
  rhi()->swapBuffers(windowRenderTarget);

  camera[0]->stop();
  delete camera[0];
#ifndef SINGLE_CAMERA
  camera[1]->stop();
  delete camera[1];
#endif

  // Release OpenGL resources
  eglMakeCurrent( demoState.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT );
  eglDestroySurface( demoState.display, demoState.surface );
  eglDestroyContext( demoState.display, demoState.context );
  eglTerminate( demoState.display );

  return 0;
}

