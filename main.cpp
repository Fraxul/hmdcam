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

struct CamTexturedQuadUniformBlock {
  glm::mat4 modelViewProjection;
};
FxAtomicString ksCamTexturedQuadUniformBlock("CamTexturedQuadUniformBlock");
RHIRenderPipeline::ptr camTexturedQuadPipeline;
RHIRenderPipeline::ptr camInvDistortionPipeline;

FxAtomicString ksImageTex("imageTex");
FxAtomicString ksDistortionMap("distortionMap");

// per-eye render targets (pre distortion)
RHISurface::ptr eyeTex[2];
RHIRenderTarget::ptr eyeRT[2];


typedef struct
{
  // HMD info/state
  ohmd_context* hmdContext;
  ohmd_device* hmdDevice;

  int hmd_width, hmd_height;
  int eye_width, eye_height;
  float ipd;
  float viewport_scale[2];
  float distortion_coeffs[4];
  float aberr_scale[3];
  float sep;
  float left_lens_center[2];
  float right_lens_center[2];
  float warp_scale;
  float warp_adj;

  glm::mat4 eyeProjection[2];

  bool rotate_screen;

  GLuint hmdDistortionProgram;
  GLint hmdDistortionProgram_coordsAttr;
  GLint hmdDistortionProgram_mvpUniform;
  GLint hmdDistortionProgram_warpTextureUniform;
  GLint hmdDistortionProgram_lensCenterUniform;
  GLint hmdDistortionProgram_viewportScaleUniform;
  GLint hmdDistortionProgram_warpScaleUniform;
  GLint hmdDistortionProgram_hmdWarpParamUniform;
  GLint hmdDistortionProgram_aberrUniform;

  int verbose;

} CUBE_STATE_T;

NvGlDemoOptions demoOptions;

static CUBE_STATE_T state;

static const GLfloat s_identityMatrix[] = {
  1.0f, 0.0f, 0.0f, 0.0f,
  0.0f, 1.0f, 0.0f, 0.0f,
  0.0f, 0.0f, 1.0f, 0.0f,
  0.0f, 0.0f, 0.0f, 1.0f
};

// vec4 out = vec4((in.xy * 2.0f) - vec2(1.0f), in.z, in.w)
static const GLfloat s_texcoordToViewportMatrix[] = {
   2.0f,  0.0f, 0.0f, 0.0f,
   0.0f,  2.0f, 0.0f, 0.0f,
   0.0f,  0.0f, 1.0f, 0.0f,
  -1.0f, -1.0f, 0.0f, 1.0f
};
// Same as above, but rotated (swapping x and y)
static const GLfloat s_texcoordToViewportMatrixRotated[] = {
   0.0f,  2.0f, 0.0f, 0.0f,
  -2.0f,  0.0f, 0.0f, 0.0f,
   0.0f,  0.0f, 1.0f, 0.0f,
   1.0f, -1.0f, 0.0f, 1.0f
};

static const GLfloat quadx[4*3] = {
   -1.0f, -1.0f,  0.0f,
    1.0f, -1.0f,  0.0f,
   -1.0f,  1.0f,  0.0f,
    1.0f,  1.0f,  0.0f,
};

static const GLfloat quadx_left[4*3] = {
   -1.0f, -1.0f,  0.0f,
    0.0f, -1.0f,  0.0f,
   -1.0f,  1.0f,  0.0f,
    0.0f,  1.0f,  0.0f,
};
static const GLfloat quadx_right[4*3] = {
    0.0f, -1.0f,  0.0f,
    1.0f, -1.0f,  0.0f,
    0.0f,  1.0f,  0.0f,
    1.0f,  1.0f,  0.0f,
};
static const GLfloat quadx_left_rotated[4*3] = {
   -1.0f, -1.0f,  0.0f,
    1.0f, -1.0f,  0.0f,
   -1.0f,  0.0f,  0.0f,
    1.0f,  0.0f,  0.0f,
};
static const GLfloat quadx_right_rotated[4*3] = {
   -1.0f,  0.0f,  0.0f,
    1.0f,  0.0f,  0.0f,
   -1.0f,  1.0f,  0.0f,
    1.0f,  1.0f,  0.0f,
};

/** Texture coordinates for the quad. */
static const GLfloat texCoords[4 * 2] = {
   0.f,  0.f,
   1.f,  0.f,
   0.f,  1.f,
   1.f,  1.f,
};
static const GLfloat texCoords_rotated[4 * 2] = {
   0.f,  1.f,
   0.f,  0.f,
   1.f,  1.f,
   1.f,  0.f,
};


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
    "shaders/camTexturedQuad.vtx.glsl",
    "shaders/camTexturedQuad.frag.glsl",
    ndcQuadVertexLayout)),
    tristripPipelineDescriptor);

  camInvDistortionPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(RHIShaderDescriptor(
    "shaders/camTexturedQuad.vtx.glsl",
    "shaders/camInvDistortion.frag.glsl",
    ndcQuadVertexLayout)),
    tristripPipelineDescriptor);


  {
    const char *vertex, *fragment;
    ohmd_gets(OHMD_GLSL_ES_DISTORTION_VERT_SRC, &vertex);
    ohmd_gets(OHMD_GLSL_ES_DISTORTION_FRAG_SRC, &fragment);
    state.hmdDistortionProgram = compileShader(vertex, fragment);
  }
  state.hmdDistortionProgram_coordsAttr = glGetAttribLocation(state.hmdDistortionProgram, "coords"); // vec2 coords
  state.hmdDistortionProgram_mvpUniform = glGetUniformLocation(state.hmdDistortionProgram, "mvp"); // model-view-projection matrix
  state.hmdDistortionProgram_warpTextureUniform = glGetUniformLocation(state.hmdDistortionProgram, "warpTexture"); // per eye texture to warp for lens distortion
  state.hmdDistortionProgram_lensCenterUniform = glGetUniformLocation(state.hmdDistortionProgram, "LensCenter"); // Position of lens center in m (usually eye_w/2, eye_h/2)
  state.hmdDistortionProgram_viewportScaleUniform = glGetUniformLocation(state.hmdDistortionProgram, "ViewportScale"); // Scale from texture co-ords to m (usually eye_w, eye_h)
  state.hmdDistortionProgram_warpScaleUniform = glGetUniformLocation(state.hmdDistortionProgram, "WarpScale"); // Distortion overall scale in m (usually ~eye_w/2)
  state.hmdDistortionProgram_hmdWarpParamUniform = glGetUniformLocation(state.hmdDistortionProgram, "HmdWarpParam"); // Distoriton coefficients (PanoTools model) [a,b,c,d]
  state.hmdDistortionProgram_aberrUniform = glGetUniformLocation(state.hmdDistortionProgram, "aberr"); // chromatic distortion post scaling
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

    ohmd_device_geti(state.hmdDevice, OHMD_SCREEN_HORIZONTAL_RESOLUTION, &state.hmd_width);
    ohmd_device_geti(state.hmdDevice, OHMD_SCREEN_VERTICAL_RESOLUTION, &state.hmd_height);

    ohmd_device_getf(state.hmdDevice, OHMD_EYE_IPD, &state.ipd);
    //viewport is half the screen
    ohmd_device_getf(state.hmdDevice, OHMD_SCREEN_HORIZONTAL_SIZE, &(state.viewport_scale[0]));
    state.viewport_scale[0] /= 2.0f;
    ohmd_device_getf(state.hmdDevice, OHMD_SCREEN_VERTICAL_SIZE, &(state.viewport_scale[1]));
    //distortion coefficients
    ohmd_device_getf(state.hmdDevice, OHMD_UNIVERSAL_DISTORTION_K, &(state.distortion_coeffs[0]));
    ohmd_device_getf(state.hmdDevice, OHMD_UNIVERSAL_ABERRATION_K, &(state.aberr_scale[0]));
    //calculate lens centers (assuming the eye separation is the distance between the lens centers)
    ohmd_device_getf(state.hmdDevice, OHMD_LENS_HORIZONTAL_SEPARATION, &state.sep);
    ohmd_device_getf(state.hmdDevice, OHMD_LENS_VERTICAL_POSITION, &(state.left_lens_center[1]));
    ohmd_device_getf(state.hmdDevice, OHMD_LENS_VERTICAL_POSITION, &(state.right_lens_center[1]));
    state.left_lens_center[0] = state.viewport_scale[0] - state.sep/2.0f;
    state.right_lens_center[0] = state.sep/2.0f;
    //assume calibration was for lens view to which ever edge of screen is further away from lens center
    state.warp_scale = (state.left_lens_center[0] > state.right_lens_center[0]) ? state.left_lens_center[0] : state.right_lens_center[0];
    state.warp_adj = 1.0f;

    state.eye_width = state.hmd_width / 2;
    state.eye_height = state.hmd_height;


    // Setup projection matrices
    ohmd_device_getf(state.hmdDevice, OHMD_LEFT_EYE_GL_PROJECTION_MATRIX, &(state.eyeProjection[0][0][0]));
    ohmd_device_getf(state.hmdDevice, OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX, &(state.eyeProjection[1][0][0]));
    // Cook the stereo separation transform into the projection matrices
    // TODO stereo separation scale
    state.eyeProjection[0] = state.eyeProjection[0] * glm::translate(glm::vec3(state.ipd *  10.0f, 0.0f, 0.0f));
    state.eyeProjection[1] = state.eyeProjection[1] * glm::translate(glm::vec3(state.ipd * -10.0f, 0.0f, 0.0f));

    ohmd_device_settings_destroy(hmdSettings);

    printf("HMD dimensions: %u x %u\n", state.hmd_width, state.hmd_height);
  }

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

      CamTexturedQuadUniformBlock ub;
      ub.modelViewProjection = mvp;
      rhi()->loadUniformBlockImmediate(ksCamTexturedQuadUniformBlock, &ub, sizeof(CamTexturedQuadUniformBlock));

      rhi()->drawNDCQuad();
#endif

      rhi()->endRenderPass(eyeRT[eyeIndex]);
    }


    // Switch to output framebuffer
    rhi()->setClearColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    rhi()->beginRenderPass(windowRenderTarget, kLoadClear);

    // XXX TODO
#if 1
    rhi()->blitTex(eyeTex[0], 0);
#else

    // Run distortion passes
    for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {

      if (state.rotate_screen) {
        if (eyeIndex == 0) {
          GL(glViewport(0, 0, state.screen_width, state.screen_height/2));
        } else {
          GL(glViewport(0, state.screen_height/2, state.screen_width, state.screen_height/2));
        }
      } else {
        if (eyeIndex == 0) {
          GL(glViewport(0, 0, state.screen_width/2, state.screen_height));
        } else {
          GL(glViewport(state.screen_width/2, 0, state.screen_width/2, state.screen_height));
        }
      }

      // Draw using distortion program
      GL(glUseProgram(state.hmdDistortionProgram));

      GL(glActiveTexture(GL_TEXTURE0));
      GL(glBindTexture(GL_TEXTURE_2D, state.eyeColorTex[eyeIndex]));
      GL(glUniform1i(state.hmdDistortionProgram_warpTextureUniform, 0)); // GL_TEXTURE0

      GL(glUniformMatrix4fv(state.hmdDistortionProgram_mvpUniform, 1, GL_FALSE, state.rotate_screen ? s_texcoordToViewportMatrixRotated : s_texcoordToViewportMatrix));
      GL(glUniform2fv(state.hmdDistortionProgram_lensCenterUniform, 1, eyeIndex == 0 ? state.left_lens_center : state.right_lens_center));
      GL(glUniform2fv(state.hmdDistortionProgram_viewportScaleUniform, 1, state.viewport_scale));
      GL(glUniform1f(state.hmdDistortionProgram_warpScaleUniform, state.warp_scale * state.warp_adj));
      GL(glUniform4fv(state.hmdDistortionProgram_hmdWarpParamUniform, 1, state.distortion_coeffs));
      GL(glUniform3fv(state.hmdDistortionProgram_aberrUniform, 1, state.aberr_scale));

      glVertexAttribPointer(state.hmdDistortionProgram_coordsAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoords);
      glEnableVertexAttribArray(state.hmdDistortionProgram_coordsAttr);
      GL(glDrawArrays( GL_TRIANGLE_STRIP, 0, 4));
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

