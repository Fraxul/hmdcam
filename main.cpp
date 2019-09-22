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

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include "EGL/egl.h"
#include "EGL/eglext.h"
#include "nvgldemo.h"

#include "ArgusCamera.h"
#include "GLUtils.h"

#include "openhmd/openhmd.h"

#ifndef M_PI
   #define M_PI 3.141592654
#endif

#define SINGLE_CAMERA

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

  int screen_width, screen_height;
  bool rotate_screen;

  GLuint camTexturedQuadProgram;
  GLint camTexturedQuadProgram_positionAttr;
  GLint camTexturedQuadProgram_texcoordAttr;
  GLint camTexturedQuadProgram_mvpUniform;
  GLint camTexturedQuadProgram_textureUniform;

  GLuint hmdDistortionProgram;
  GLint hmdDistortionProgram_coordsAttr;
  GLint hmdDistortionProgram_mvpUniform;
  GLint hmdDistortionProgram_warpTextureUniform;
  GLint hmdDistortionProgram_lensCenterUniform;
  GLint hmdDistortionProgram_viewportScaleUniform;
  GLint hmdDistortionProgram_warpScaleUniform;
  GLint hmdDistortionProgram_hmdWarpParamUniform;
  GLint hmdDistortionProgram_aberrUniform;

  GLuint eyeFBO[2];
  GLuint eyeColorTex[2];

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

  // TODO
  state.screen_width = demoOptions.windowSize[0];
  state.screen_height = demoOptions.windowSize[1];



  glViewport(0, 0, (GLsizei)state.screen_width, (GLsizei)state.screen_height);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);

  // Set up shared resources

  state.camTexturedQuadProgram = compileShader(
    "attribute vec4 vPosition; \n"
    "attribute vec2 TexCoordIn; \n"
    "varying vec2 TexCoordOut; \n"
    "uniform mat4 modelViewProjection; \n"
    "void main() { \n"
    "  gl_Position = modelViewProjection * vPosition; \n"
    "  TexCoordOut = TexCoordIn; \n"
    "} \n",

    "#extension GL_OES_EGL_image_external : require\n"
    "precision highp float;"
    "varying vec2 TexCoordOut;"
    "uniform samplerExternalOES Texture;"
    "void main() { \n"
    "  gl_FragColor = texture2D(Texture, TexCoordOut); \n"
    "} \n"
  );

  state.camTexturedQuadProgram_positionAttr = glGetAttribLocation(state.camTexturedQuadProgram, "vPosition");
  state.camTexturedQuadProgram_texcoordAttr = glGetAttribLocation(state.camTexturedQuadProgram, "TexCoordIn");
	state.camTexturedQuadProgram_mvpUniform = glGetUniformLocation(state.camTexturedQuadProgram, "modelViewProjection");
	state.camTexturedQuadProgram_textureUniform = glGetUniformLocation(state.camTexturedQuadProgram, "Texture");

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

  init_ogl();

  // Create FBOs for per-eye rendering (pre distortion)
  createFBO(state.eye_width, state.eye_height, &state.eyeFBO[0], &state.eyeColorTex[0]);
  createFBO(state.eye_width, state.eye_height, &state.eyeFBO[1], &state.eyeColorTex[1]);

  printf("Screen dimensions: %u x %u\n", state.screen_width, state.screen_height);

  if (state.screen_width == state.hmd_width && state.screen_height == state.hmd_height) {
    // Screen physical orientation matches HMD logical orientation
  } else if (state.screen_width == state.hmd_height && state.screen_height == state.hmd_width) {
    // Screen is oriented opposite of HMD logical orientation
    state.rotate_screen = true;
    printf("Will compensate for screen rotation.\n");
  } else {
    printf("WARNING: Screen and HMD dimensions don't match; check system configuration.\n");
  }

  ArgusCamera* leftCamera = new ArgusCamera(demoState.display, demoState.context, 0, 1280, 720);
#ifndef SINGLE_CAMERA
  ArgusCamera* rightCamera = new ArgusCamera(demoState.display, demoState.context, 1, 1280, 720);
#endif

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);

  while (!want_quit)
  {
    leftCamera->readFrame();
#ifndef SINGLE_CAMERA
    rightCamera->readFrame();
#endif


    for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
#ifdef SINGLE_CAMERA
      ArgusCamera* activeCamera = leftCamera;
#else
      ArgusCamera* activeCamera = (eyeIndex == 0) ? leftCamera : rightCamera;
#endif
      // Target eye FBO
      GL(glBindFramebuffer(GL_FRAMEBUFFER, state.eyeFBO[eyeIndex]));
      GL(glViewport(0, 0, state.eye_width, state.eye_height));
      glClear(GL_COLOR_BUFFER_BIT);

      // Draw camera content for thie eye
      glUseProgram(state.camTexturedQuadProgram);
      GL(glActiveTexture(GL_TEXTURE0));
      GL(glBindTexture(GL_TEXTURE_EXTERNAL_OES, activeCamera->rgbTexture()));
      glUniform1i(state.camTexturedQuadProgram_textureUniform, 0);
      // coordsys right now: -X = left, -Z = into screen
      // (camera is at the origin)
      const glm::vec3 tx = glm::vec3(0.0f, 0.0f, -7.0f);
      const float scaleFactor = 4.0f;
      glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(-scaleFactor * (static_cast<float>(activeCamera->streamWidth()) / static_cast<float>(activeCamera->streamHeight())), scaleFactor, 1.0f)); // TODO
      glm::mat4 mvp = state.eyeProjection[eyeIndex] * model;
      glUniformMatrix4fv(state.camTexturedQuadProgram_mvpUniform, 1, GL_FALSE, &mvp[0][0]);

      glVertexAttribPointer(state.camTexturedQuadProgram_positionAttr, 3, GL_FLOAT, GL_FALSE, 0, quadx );
      glVertexAttribPointer(state.camTexturedQuadProgram_texcoordAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoords );
      glEnableVertexAttribArray(state.camTexturedQuadProgram_positionAttr);
      glEnableVertexAttribArray(state.camTexturedQuadProgram_texcoordAttr);

      GL(glDrawArrays( GL_TRIANGLE_STRIP, 0, 4));
    }


    // Switch to output framebuffer
    GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));

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

      GL(glUniformMatrix4fv(state.hmdDistortionProgram_mvpUniform, 1, GL_FALSE, state.rotate_screen ? s_texcoordToViewportMatrixRotated : s_texcoordToViewportMatrix));
      GL(glUniform1i(state.hmdDistortionProgram_warpTextureUniform, 0)); // GL_TEXTURE0
      GL(glUniform2fv(state.hmdDistortionProgram_lensCenterUniform, 1, eyeIndex == 0 ? state.left_lens_center : state.right_lens_center));
      GL(glUniform2fv(state.hmdDistortionProgram_viewportScaleUniform, 1, state.viewport_scale));
      GL(glUniform1f(state.hmdDistortionProgram_warpScaleUniform, state.warp_scale * state.warp_adj));
      GL(glUniform4fv(state.hmdDistortionProgram_hmdWarpParamUniform, 1, state.distortion_coeffs));
      GL(glUniform3fv(state.hmdDistortionProgram_aberrUniform, 1, state.aberr_scale));

      glVertexAttribPointer(state.hmdDistortionProgram_coordsAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoords);
      glEnableVertexAttribArray(state.hmdDistortionProgram_coordsAttr);
      GL(glDrawArrays( GL_TRIANGLE_STRIP, 0, 4));
    }

    eglSwapBuffers(demoState.display, demoState.surface);
    update_fps();
  }

  // Restore signal handlers
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);

  // clear screen
  glClear( GL_COLOR_BUFFER_BIT );
  eglSwapBuffers(demoState.display, demoState.surface);

  leftCamera->stop();
  delete leftCamera;
#ifndef SINGLE_CAMERA
  rightCamera->stop();
  delete rightCamera;
#endif

  // Release OpenGL resources
  eglMakeCurrent( demoState.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT );
  eglDestroySurface( demoState.display, demoState.surface );
  eglDestroyContext( demoState.display, demoState.context );
  eglTerminate( demoState.display );

  return 0;
}

