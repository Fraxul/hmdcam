#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

#include "bcm_host.h"

#include <GLES/gl.h>
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

typedef struct
{
  uint32_t screen_width;
  uint32_t screen_height;
  // OpenGL|ES objects
  EGLDisplay display;
  EGLSurface surface;
  EGLContext context;

  GLuint camTexturedQuadProgram;
  GLint camTexturedQuadProgram_positionAttr;
  GLint camTexturedQuadProgram_texcoordAttr;
  GLint camTexturedQuadProgram_textureUniform;

  int verbose;

} CUBE_STATE_T;


static void init_ogl();
static CUBE_STATE_T state;

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

   // Set background color and clear buffers
   glViewport(0, 0, (GLsizei)state.screen_width, (GLsizei)state.screen_height);
   glClearColor(0.15f, 0.25f, 0.35f, 1.0f);

   glDisable(GL_CULL_FACE);
   glDisable(GL_DEPTH_TEST);
}

static void init_shaders() {

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

}




static bool want_quit = false;
static void signal_handler(int) {
  want_quit = true;
}

int main(int argc, char* argv[]) {

  bcm_host_init();

  init_ogl();
  printf("Screen dimensions: %u x %u\n", state.screen_width, state.screen_height);

  init_shaders();

  MMALCamera* leftCamera = new MMALCamera(state.display, state.context);
  MMALCamera* rightCamera = new MMALCamera(state.display, state.context);
  leftCamera->init(/*cameraIndex=*/0, 1280, 720, 30);
  rightCamera->init(/*cameraIndex=*/1, 1280, 720, 30);

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);

  while (!want_quit)
  {
    leftCamera->readFrame();
    rightCamera->readFrame();

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

    eglSwapBuffers(state.display, state.surface);
  }

  // Restore signal handlers
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);

  // clear screen
  glClear( GL_COLOR_BUFFER_BIT );
  eglSwapBuffers(state.display, state.surface);

  leftCamera->stop();
  rightCamera->stop();
  delete leftCamera;
  delete rightCamera;

  // Release OpenGL resources
  eglMakeCurrent( state.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT );
  eglDestroySurface( state.display, state.surface );
  eglDestroyContext( state.display, state.context );
  eglTerminate( state.display );

  return 0;
}

