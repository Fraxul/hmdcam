#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#include "bcm_host.h"

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include "EGL/egl.h"
#include "EGL/eglext.h"

#include "GLUtils.h"
#include "MMALCamera.h"

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

  uint32_t screen_width, screen_height;
  bool rotate_screen;
  // OpenGL|ES objects
  EGLDisplay display;
  EGLSurface surface;
  EGLContext context;

  GLuint camTexturedQuadProgram;
  GLint camTexturedQuadProgram_positionAttr;
  GLint camTexturedQuadProgram_texcoordAttr;
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

  GLuint eyeFBO;
  GLuint eyeColorTex;

  int verbose;

} CUBE_STATE_T;


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

/** Texture coordinates for the quad. */
static const GLfloat texCoords[4 * 2] = {
   0.f,  0.f,
   1.f,  0.f,
   0.f,  1.f,
   1.f,  1.f,
};


static void init_ogl() {
   int32_t success = 0;
   EGLBoolean result;
   EGLint num_config;

   static EGL_DISPMANX_WINDOW_T nativewindow;

   DISPMANX_ELEMENT_HANDLE_T dispman_element;
   DISPMANX_DISPLAY_HANDLE_T dispman_display;
   DISPMANX_UPDATE_HANDLE_T dispman_update;
   VC_RECT_T dst_rect;
   VC_RECT_T src_rect;

   static const EGLint attribute_list[] =
   {
      EGL_RED_SIZE, 8,
      EGL_GREEN_SIZE, 8,
      EGL_BLUE_SIZE, 8,
      EGL_ALPHA_SIZE, 8,
      EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
      EGL_NONE
   };

   static const EGLint context_attributes[] =
   {
      EGL_CONTEXT_CLIENT_VERSION, 2,
      EGL_NONE
   };

   EGLConfig config;

   // get an EGL display connection
   state.display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
   assert(state.display!=EGL_NO_DISPLAY);

   // initialize the EGL display connection
   result = eglInitialize(state.display, NULL, NULL);
   assert(EGL_FALSE != result);

   // get an appropriate EGL frame buffer configuration
   result = eglChooseConfig(state.display, attribute_list, &config, 1, &num_config);
   assert(EGL_FALSE != result);

   // create an EGL rendering context
   state.context = eglCreateContext(state.display, config, EGL_NO_CONTEXT, context_attributes);
   assert(state.context!=EGL_NO_CONTEXT);

   // create an EGL window surface
   success = graphics_get_display_size(0 /* LCD */, &state.screen_width, &state.screen_height);
   assert( success >= 0 );

   //state.screen_width = 800;
   //state.screen_height = 600;

   dst_rect.x = 0;
   dst_rect.y = 0;
   dst_rect.width = state.screen_width;
   dst_rect.height = state.screen_height;

   src_rect.x = 0;
   src_rect.y = 0;
   src_rect.width = state.screen_width << 16;
   src_rect.height = state.screen_height << 16;

   dispman_display = vc_dispmanx_display_open( 0 /* LCD */);
   dispman_update = vc_dispmanx_update_start( 0 );

   dispman_element = vc_dispmanx_element_add ( dispman_update, dispman_display,
      0/*layer*/, &dst_rect, 0/*src*/,
      &src_rect, DISPMANX_PROTECTION_NONE, 0 /*alpha*/, 0/*clamp*/, (DISPMANX_TRANSFORM_T) 0/*transform*/);

   nativewindow.element = dispman_element;
   nativewindow.width = state.screen_width;
   nativewindow.height = state.screen_height;
   vc_dispmanx_update_submit_sync( dispman_update );

   state.surface = eglCreateWindowSurface( state.display, config, &nativewindow, NULL );
   assert(state.surface != EGL_NO_SURFACE);

   // connect the context to the surface
   result = eglMakeCurrent(state.display, state.surface, state.surface, state.context);
   assert(EGL_FALSE != result);

   glViewport(0, 0, (GLsizei)state.screen_width, (GLsizei)state.screen_height);
   glDisable(GL_CULL_FACE);
   glDisable(GL_DEPTH_TEST);

  // Set up shared resources

  state.camTexturedQuadProgram = compileShader(
    "attribute vec4 vPosition;"
    "attribute vec2 TexCoordIn;"
    "varying vec2 TexCoordOut;"
    "void main() { \n"
    "  gl_Position = vPosition; \n"
    "  TexCoordOut = TexCoordIn; \n"
    "} \n",

    "#extension GL_OES_EGL_image_external : require\n"
    "precision mediump float;"
    "varying vec2 TexCoordOut;"
    "uniform samplerExternalOES Texture;"
    "void main() { \n"
    "  gl_FragColor = texture2D(Texture, TexCoordOut); \n"
    "} \n"
  );

  state.camTexturedQuadProgram_positionAttr = glGetAttribLocation(state.camTexturedQuadProgram, "vPosition");
  state.camTexturedQuadProgram_texcoordAttr = glGetAttribLocation(state.camTexturedQuadProgram, "TexCoordIn");
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
}

int main(int argc, char* argv[]) {

  memset(&state, NULL, sizeof(state));

  bcm_host_init();

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

    ohmd_device_settings_destroy(hmdSettings);

    printf("HMD dimensions: %u x %u\n", state.hmd_width, state.hmd_height);
  }

  init_ogl();

  createFBO(state.eye_width, state.eye_height, &state.eyeFBO, &state.eyeColorTex);

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

  MMALCamera* leftCamera = new MMALCamera(state.display, state.context);
#ifndef SINGLE_CAMERA
  MMALCamera* rightCamera = new MMALCamera(state.display, state.context);
#endif

  leftCamera->init(/*cameraIndex=*/0, 640, 480);
#ifndef SINGLE_CAMERA
  rightCamera->init(/*cameraIndex=*/1, 640, 480);
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

#if 0
    // Draw
    glUseProgram(state.camTexturedQuadProgram);

    {
      // Load camera texture into unit 0
      GL(glActiveTexture(GL_TEXTURE0));
      GL(glBindTexture(GL_TEXTURE_EXTERNAL_OES, leftCamera->rgbTexture()));
      glUniform1i(state.camTexturedQuadProgram_textureUniform, 0);

      glVertexAttribPointer(state.camTexturedQuadProgram_positionAttr, 3, GL_FLOAT, GL_FALSE, 0, quadx_left );
      glVertexAttribPointer(state.camTexturedQuadProgram_texcoordAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoords );
      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);

      // draw first 4 vertices
      GL(glDrawArrays( GL_TRIANGLE_STRIP, 0, 4));
    }

    {
      // Load camera texture into unit 0
      GL(glActiveTexture(GL_TEXTURE0));
      GL(glBindTexture(GL_TEXTURE_EXTERNAL_OES, rightCamera->rgbTexture()));
      glUniform1i(state.camTexturedQuadProgram_textureUniform, 0);

      glVertexAttribPointer(state.camTexturedQuadProgram_positionAttr, 3, GL_FLOAT, GL_FALSE, 0, quadx_right );
      glVertexAttribPointer(state.camTexturedQuadProgram_texcoordAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoords );
      glEnableVertexAttribArray(0);
      glEnableVertexAttribArray(1);

      // draw first 4 vertices
      GL(glDrawArrays( GL_TRIANGLE_STRIP, 0, 4));
    }
#else


    for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
      // Target eye FBO
      GL(glBindFramebuffer(GL_FRAMEBUFFER, state.eyeFBO));
      GL(glViewport(0, 0, state.eye_width, state.eye_height));

      // Draw camera content for thie eye
      glUseProgram(state.camTexturedQuadProgram);
      GL(glActiveTexture(GL_TEXTURE0));
#ifdef SINGLE_CAMERA
      GL(glBindTexture(GL_TEXTURE_EXTERNAL_OES, leftCamera->rgbTexture()));
#else
      GL(glBindTexture(GL_TEXTURE_EXTERNAL_OES, eyeIndex == 0 ? leftCamera->rgbTexture() : rightCamera->rgbTexture()));
#endif
      glUniform1i(state.camTexturedQuadProgram_textureUniform, 0);

      glVertexAttribPointer(state.camTexturedQuadProgram_positionAttr, 3, GL_FLOAT, GL_FALSE, 0, quadx );
      glVertexAttribPointer(state.camTexturedQuadProgram_texcoordAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoords );
      glEnableVertexAttribArray(state.camTexturedQuadProgram_positionAttr);
      glEnableVertexAttribArray(state.camTexturedQuadProgram_texcoordAttr);

      GL(glDrawArrays( GL_TRIANGLE_STRIP, 0, 4));

      // Switch to output framebuffer
      GL(glBindFramebuffer(GL_FRAMEBUFFER, 0));
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

#endif

    eglSwapBuffers(state.display, state.surface);
    update_fps();
  }

  // Restore signal handlers
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);

  // clear screen
  glClear( GL_COLOR_BUFFER_BIT );
  eglSwapBuffers(state.display, state.surface);

  leftCamera->stop();
  delete leftCamera;
#ifndef SINGLE_CAMERA
  rightCamera->stop();
  delete rightCamera;
#endif

  // Release OpenGL resources
  eglMakeCurrent( state.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT );
  eglDestroySurface( state.display, state.surface );
  eglDestroyContext( state.display, state.context );
  eglTerminate( state.display );

  return 0;
}

