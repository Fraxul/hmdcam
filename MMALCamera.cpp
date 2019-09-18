#include "MMALCamera.h"

#include <bcm_host.h>
#include "interface/vcos/vcos.h"

#include "interface/mmal/mmal.h"
#include "interface/mmal/mmal_logging.h"
#include "interface/mmal/mmal_buffer.h"
#include "interface/mmal/util/mmal_util.h"
#include "interface/mmal/util/mmal_util_params.h"
#include "interface/mmal/util/mmal_default_components.h"
#include "interface/mmal/util/mmal_connection.h"

#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglext_brcm.h>
  
#include "GLUtils.h"
#include "RaspiCamControl.h"

#include <stdio.h>
  
// Standard port setting for the camera component
#define MMAL_CAMERA_PREVIEW_PORT 0
#define MMAL_CAMERA_VIDEO_PORT 1
#define MMAL_CAMERA_CAPTURE_PORT 2

/// Video render needs at least 2 buffers.
#define VIDEO_OUTPUT_BUFFERS_NUM 3

// Frames rates of 0 implies variable, but denominator needs to be 1 to prevent div by 0
#define PREVIEW_FRAME_RATE_NUM 0
#define PREVIEW_FRAME_RATE_DEN 1
#define STILLS_FRAME_RATE_NUM 0
#define STILLS_FRAME_RATE_DEN 1

#define MMAL_CHECK(x) _mmal_check(x, #x, __FILE__, __LINE__ )

void _mmal_check(MMAL_STATUS_T st, const char* op, const char* file, int line) {
  if (st != MMAL_SUCCESS) {
    fprintf(stderr, "MMAL error after %s (%s:%d): 0x%x: %s\n", op, file, line, st, mmal_status_to_string(st));
    abort();
  }
}

bool MMALCamera::readFrame() {
  // Send empty buffers to camera preview port
  MMAL_BUFFER_HEADER_T* buf;
  while ((buf = mmal_queue_get(m_previewPool->queue)) != NULL) {
     MMAL_STATUS_T st = mmal_port_send_buffer(m_previewPort, buf);
     if (st != MMAL_SUCCESS) {
        vcos_log_error("Failed to send buffer to %s", m_previewPort->name);
     }
  }

  // Process returned buffer, if present
  buf = mmal_queue_get(m_previewQueue);
  if (!buf)
    return false;

  // Update textures
#if ENABLE_RGB
  do_update_texture(EGL_IMAGE_BRCM_MULTIMEDIA,   (EGLClientBuffer) buf->data, &m_rgbTex, &m_rgbImage);
#endif
#if ENABLE_YUV
  do_update_texture(EGL_IMAGE_BRCM_MULTIMEDIA_Y, (EGLClientBuffer) buf->data, &m_yTex, &m_yImage);
  do_update_texture(EGL_IMAGE_BRCM_MULTIMEDIA_U, (EGLClientBuffer) buf->data, &m_uTex, &m_uImage);
  do_update_texture(EGL_IMAGE_BRCM_MULTIMEDIA_V, (EGLClientBuffer) buf->data, &m_vTex, &m_vImage);
#endif


  // Now return the PREVIOUS MMAL buffer header back to the camera preview.
  if (m_previewBuf)
   mmal_buffer_header_release(m_previewBuf);

  m_previewBuf = buf;

  return true; // should be OK to draw
}

void MMALCamera::do_update_texture(EGLenum target, EGLClientBuffer mm_buf, GLuint *texture, EGLImageKHR *egl_image)
{
   GL(glBindTexture(GL_TEXTURE_EXTERNAL_OES, *texture));
   if (*egl_image != EGL_NO_IMAGE_KHR) {
      // Discard the old EGL image for the preview frame
      eglDestroyImageKHR(m_eglDisplay, *egl_image);
      *egl_image = EGL_NO_IMAGE_KHR;
   }

   *egl_image = eglCreateImageKHR(m_eglDisplay, EGL_NO_CONTEXT, target, mm_buf, NULL);
   GL(glEGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, *egl_image));
}



MMALCamera::MMALCamera(EGLDisplay display, EGLContext context) :
  m_eglDisplay(display), m_eglContext(context),
#if ENABLE_YUV
  m_yTex(0), m_uTex(0), m_vTex(0), m_yImage(NULL), m_uImage(NULL), m_vImage(NULL), 
#endif
#if ENABLE_RGB
  m_rgbTex(0), m_rgbImage(NULL),
#endif
  m_cameraComponent(NULL), m_previewPort(NULL), m_previewPool(NULL), m_previewQueue(NULL), m_previewBuf(NULL) {


#if ENABLE_YUV
  GL(glGenTextures(1, &m_yTex));
  GL(glGenTextures(1, &m_uTex));
  GL(glGenTextures(1, &m_vTex));
#endif
#if ENABLE_RGB
  GL(glGenTextures(1, &m_rgbTex));
#endif
}

MMALCamera::~MMALCamera() {

}

void MMALCamera::init(unsigned int cameraIndex, unsigned int Width, unsigned int Height) {
   /* Create the component */
   MMAL_CHECK(mmal_component_create(MMAL_COMPONENT_DEFAULT_CAMERA, &m_cameraComponent));

#if 0
   MMAL_CHECK(raspicamcontrol_set_stereo_mode(m_cameraComponent->output[0], cameraParameters.stereo_mode));
   MMAL_CHECK(raspicamcontrol_set_stereo_mode(m_cameraComponent->output[1], cameraParameters.stereo_mode));
   MMAL_CHECK(raspicamcontrol_set_stereo_mode(m_cameraComponent->output[2], cameraParameters.stereo_mode));
#endif

   MMAL_PARAMETER_INT32_T camera_num = {{MMAL_PARAMETER_CAMERA_NUM, sizeof(camera_num)}, cameraIndex};
   MMAL_CHECK(mmal_port_parameter_set(m_cameraComponent->control, &camera_num.hdr));

   if (!m_cameraComponent->output_num)
      die("Camera doesn't have output ports");

   MMAL_CHECK(mmal_port_parameter_set_uint32(m_cameraComponent->control, MMAL_PARAMETER_CAMERA_CUSTOM_SENSOR_CONFIG, 0 /*state->common_settings.sensor_mode*/));

   m_previewPort = m_cameraComponent->output[MMAL_CAMERA_PREVIEW_PORT];
   MMAL_PORT_T* video_port = m_cameraComponent->output[MMAL_CAMERA_VIDEO_PORT];
   MMAL_PORT_T* still_port = m_cameraComponent->output[MMAL_CAMERA_CAPTURE_PORT];

   // Enable the camera, and tell it its control callback function
   m_cameraComponent->control->userdata = reinterpret_cast<struct MMAL_PORT_USERDATA_T *>(this);
   MMAL_CHECK(mmal_port_enable(m_cameraComponent->control, &cameraControlCallback_thunk));

   //  set up the camera configuration
   {
      MMAL_PARAMETER_CAMERA_CONFIG_T cam_config =
      {
         { MMAL_PARAMETER_CAMERA_CONFIG, sizeof(cam_config) },
         .max_stills_w = Width,
         .max_stills_h = Height,
         .stills_yuv422 = 0,
         .one_shot_stills = 1,
         .max_preview_video_w = Width,
         .max_preview_video_h = Height,
         .num_preview_video_frames = 3,
         .stills_capture_circular_buffer_height = 0,
         .fast_preview_resume = 0,
         .use_stc_timestamp = MMAL_PARAM_TIMESTAMP_MODE_RESET_STC
      };
/*
      if (state->fullResPreview)
      {
         cam_config.max_preview_video_w = state->common_settings.width;
         cam_config.max_preview_video_h = state->common_settings.height;
      }
*/

      mmal_port_parameter_set(m_cameraComponent->control, &cam_config.hdr);
   }

  RASPICAM_CAMERA_PARAMETERS cameraParameters;
  raspicamcontrol_set_defaults(&cameraParameters);

  raspicamcontrol_set_all_parameters(m_cameraComponent, &cameraParameters);

   // Now set up the port formats

   MMAL_ES_FORMAT_T* format = m_previewPort->format;
   format->encoding = MMAL_ENCODING_OPAQUE;
   format->encoding_variant = MMAL_ENCODING_I420;

   if(cameraParameters.shutter_speed > 6000000)
   {
      MMAL_PARAMETER_FPS_RANGE_T fps_range = {{MMAL_PARAMETER_FPS_RANGE, sizeof(fps_range)},
         { 50, 1000 }, {166, 1000}
      };
      mmal_port_parameter_set(m_previewPort, &fps_range.hdr);
   }
   else if(cameraParameters.shutter_speed > 1000000)
   {
      MMAL_PARAMETER_FPS_RANGE_T fps_range = {{MMAL_PARAMETER_FPS_RANGE, sizeof(fps_range)},
         { 166, 1000 }, {999, 1000}
      };
      mmal_port_parameter_set(m_previewPort, &fps_range.hdr);
   }
/*
   if (state->fullResPreview)
   {
      // In this mode we are forcing the preview to be generated from the full capture resolution.
      // This runs at a max of 15fps with the OV5647 sensor.
      format->es->video.width = VCOS_ALIGN_UP(state->common_settings.width, 32);
      format->es->video.height = VCOS_ALIGN_UP(state->common_settings.height, 16);
      format->es->video.crop.x = 0;
      format->es->video.crop.y = 0;
      format->es->video.crop.width = state->common_settings.width;
      format->es->video.crop.height = state->common_settings.height;
      format->es->video.frame_rate.num = FULL_RES_PREVIEW_FRAME_RATE_NUM;
      format->es->video.frame_rate.den = FULL_RES_PREVIEW_FRAME_RATE_DEN;
   }
   else
*/
   {
      // Use a full FOV 4:3 mode
      format->es->video.width = VCOS_ALIGN_UP(Width, 32);
      format->es->video.height = VCOS_ALIGN_UP(Height, 16);
      format->es->video.crop.x = 0;
      format->es->video.crop.y = 0;
      format->es->video.crop.width = Width;
      format->es->video.crop.height = Height;
      format->es->video.frame_rate.num = PREVIEW_FRAME_RATE_NUM;
      format->es->video.frame_rate.den = PREVIEW_FRAME_RATE_DEN;
   }

   MMAL_CHECK(mmal_port_format_commit(m_previewPort));

   // Set the same format on the video  port (which we don't use here)
   mmal_format_full_copy(video_port->format, format);
   MMAL_CHECK(mmal_port_format_commit(video_port));

   // Ensure there are enough buffers to avoid dropping frames
   if (video_port->buffer_num < VIDEO_OUTPUT_BUFFERS_NUM)
      video_port->buffer_num = VIDEO_OUTPUT_BUFFERS_NUM;

   format = still_port->format;

   if(cameraParameters.shutter_speed > 6000000)
   {
      MMAL_PARAMETER_FPS_RANGE_T fps_range = {{MMAL_PARAMETER_FPS_RANGE, sizeof(fps_range)},
         { 50, 1000 }, {166, 1000}
      };
      mmal_port_parameter_set(still_port, &fps_range.hdr);
   }
   else if(cameraParameters.shutter_speed > 1000000)
   {
      MMAL_PARAMETER_FPS_RANGE_T fps_range = {{MMAL_PARAMETER_FPS_RANGE, sizeof(fps_range)},
         { 167, 1000 }, {999, 1000}
      };
      mmal_port_parameter_set(still_port, &fps_range.hdr);
   }
   // Set our stills format on the stills (for encoder) port
   format->encoding = MMAL_ENCODING_OPAQUE;
   format->es->video.width = VCOS_ALIGN_UP(Width, 32);
   format->es->video.height = VCOS_ALIGN_UP(Height, 16);
   format->es->video.crop.x = 0;
   format->es->video.crop.y = 0;
   format->es->video.crop.width = Width;
   format->es->video.crop.height = Height;
   format->es->video.frame_rate.num = STILLS_FRAME_RATE_NUM;
   format->es->video.frame_rate.den = STILLS_FRAME_RATE_DEN;

   MMAL_CHECK(mmal_port_format_commit(still_port));

   /* Ensure there are enough buffers to avoid dropping frames */
   if (still_port->buffer_num < VIDEO_OUTPUT_BUFFERS_NUM)
      still_port->buffer_num = VIDEO_OUTPUT_BUFFERS_NUM;

   /* Enable component */
   MMAL_CHECK(mmal_component_enable(m_cameraComponent));

   /* Enable ZERO_COPY mode on the preview port which instructs MMAL to only
    * pass the 4-byte opaque buffer handle instead of the contents of the opaque
    * buffer.
    * The opaque handle is resolved on VideoCore by the GL driver when the EGL
    * image is created.
    */
   MMAL_CHECK(mmal_port_parameter_set_boolean(m_previewPort, MMAL_PARAMETER_ZERO_COPY, MMAL_TRUE));

   MMAL_CHECK(mmal_port_format_commit(m_previewPort));

   /* For GL a pool of opaque buffer handles must be allocated in the client.
    * These buffers are used to create the EGL images.
    */
   m_previewPort->buffer_num = m_previewPort->buffer_num_recommended;
   m_previewPort->buffer_size = m_previewPort->buffer_size_recommended;

   fprintf(stderr, "Creating buffer pool for GL renderer num %d size %d\n",
                  m_previewPort->buffer_num, m_previewPort->buffer_size);

   /* Pool + queue to hold preview frames */
   m_previewPool = mmal_port_pool_create(m_previewPort, m_previewPort->buffer_num, m_previewPort->buffer_size);
   if (!m_previewPool)
      die("Error allocating preview pool");

   /* Place filled buffers from the preview port in a queue to render */
   m_previewQueue = mmal_queue_create();
   if (!m_previewQueue)
      die("Error allocating queue");

   /* Enable preview port callback */
   m_previewPort->userdata = reinterpret_cast<struct MMAL_PORT_USERDATA_T*>(this);
   MMAL_CHECK(mmal_port_enable(m_previewPort, &previewOutputCallback_thunk));
}

/*static*/ void MMALCamera::cameraControlCallback_thunk(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer){
  reinterpret_cast<MMALCamera*>(port->userdata)->controlCallback(port, buffer);
}

void MMALCamera::controlCallback(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer) {
  printf("Camera control callback\n\n");
  return;
}


/*static*/ void MMALCamera::previewOutputCallback_thunk(MMAL_PORT_T *port, MMAL_BUFFER_HEADER_T *buffer){
  reinterpret_cast<MMALCamera*>(port->userdata)->previewOutputCallback(port, buffer);
}

void MMALCamera::previewOutputCallback(MMAL_PORT_T* port, MMAL_BUFFER_HEADER_T* buf){
   if (buf->length == 0) {
      fprintf(stderr, "%s: zero-length buffer => EOS\n", port->name);
      //state->preview_stop = 1;
      mmal_buffer_header_release(buf);
   } else if (buf->data == NULL) {
      fprintf(stderr, "%s: zero buffer handle\n", port->name);
      mmal_buffer_header_release(buf);
   } else {
      /* Enqueue the preview frame for rendering and return to
       * avoid blocking MMAL core.
       */
      mmal_queue_put(m_previewQueue, buf);
   }
}

void MMALCamera::stop() {

}

