#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/chrono/duration.hpp>
#include <boost/chrono/system_clocks.hpp>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/gl/RHIEGLSurfaceRenderTargetGL.h"
#include "rhi/gl/GLCommon.h"

#include "nvgldemo.h"

#include "ArgusCamera.h"
#include "InputListener.h"

#include "imgui.h"
#include "imgui_backend.h"

#include "openhmd/openhmd.h"

#define STBI_ONLY_PNG
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include "Calibration.h"

// Camera config
// Size parameters for sensor mode selection.
// Note that changing the sensor mode will invalidate the calibration
// (Pixel coordinates are baked into the calibration data)
size_t s_cameraWidth = 1280, s_cameraHeight = 720;
//const size_t s_cameraWidth = 1920, s_cameraHeight = 1080;

// Requested capture rate for the camera. This should be the framerate of the display device, with as much precision as possible.
// TODO: autodetect this. (current value pulled from running `fbset`)
const double s_cameraFramerate = 89.527;

// #define SWAP_CAMERA_EYES
#define CAMERA_INVERTED 1 // 0 = upright, 1 = camera rotated 180 degrees. (90 degree rotation is not supported)

// Mapping of libargus camera device ID to index 0 (left) and 1 (right).
#ifdef SWAP_CAMERA_EYES
  #define LEFT_CAMERA_INDEX 1
  #define RIGHT_CAMERA_INDEX 0
#else
  #define LEFT_CAMERA_INDEX 0
  #define RIGHT_CAMERA_INDEX 1
#endif

RHIRenderTarget::ptr windowRenderTarget;

struct NDCQuadUniformBlock {
  glm::mat4 modelViewProjection;
};
FxAtomicString ksNDCQuadUniformBlock("NDCQuadUniformBlock");

struct NDCClippedQuadUniformBlock {
  glm::mat4 modelViewProjection;
  glm::vec2 minUV;
  glm::vec2 maxUV;
};
FxAtomicString ksNDCClippedQuadUniformBlock("NDCClippedQuadUniformBlock");
RHIRenderPipeline::ptr camTexturedQuadPipeline;
RHIRenderPipeline::ptr camOverlayPipeline;
RHIRenderPipeline::ptr camOverlayStereoPipeline;
RHIRenderPipeline::ptr camOverlayStereoUndistortPipeline;
RHIRenderPipeline::ptr camUndistortMaskPipeline;
RHIRenderPipeline::ptr camGreyscalePipeline;
RHIRenderPipeline::ptr camGreyscaleUndistortPipeline;

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
int hmd_width, hmd_height;
int eye_width, eye_height;
bool rotate_screen = false;
glm::mat4 eyeProjection[2];
glm::mat4 eyeView[2];

float stereoSeparationScale = 2.0f; // TODO: why? some of the Vive projection math is still wrong...

// Camera render parameters
float scaleFactor = 2.1f;
float stereoOffset = 0.0f;

// Camera info/state
ArgusCamera* stereoCamera;
RHISurface::ptr cameraDistortionMap[2];
RHISurface::ptr cameraMask[2];

static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

NvGlDemoOptions demoOptions;

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

  viveDistortionPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(RHIShaderDescriptor(
    "shaders/hmdDistortion.vtx.glsl",
    "shaders/viveDistortion.frag.glsl",
    ndcQuadVertexLayout)),
    tristripPipelineDescriptor);

}

bool want_quit = false;
static void signal_handler(int) {
  want_quit = true;

  // Restore signal handlers so the program is still interruptable if clean shutdown gets stuck
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);
}

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

int main(int argc, char* argv[]) {

  startInputListenerThread();
  init_ogl();

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Keyboard navigation (mapped from the InputListener media remote interface)
  // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad; // Gamepad navigation (not used right now)
  ImGui_ImplOpenGL3_Init(NULL);
  ImGui_ImplInputListener_Init();

  io.DisplaySize = ImVec2(512.0f, 512.0f);
  io.DisplayFramebufferScale = ImVec2(2.0f, 2.0f);


  hmdContext = ohmd_ctx_create();
  int num_devices = ohmd_ctx_probe(hmdContext);
  if (num_devices < 0){
    printf("OpenHMD: failed to probe devices: %s\n", ohmd_ctx_get_error(hmdContext));
    return 1;
  }

  {
    ohmd_device_settings* hmdSettings = ohmd_device_settings_create(hmdContext);

    hmdDevice = ohmd_list_open_device_s(hmdContext, 0, hmdSettings);
    if (!hmdDevice){
      printf("OpenHMD: failed to open device: %s\n", ohmd_ctx_get_error(hmdContext));
      return 1;
    }

    // Not used after ohmd_list_open_device_s returns
    ohmd_device_settings_destroy(hmdSettings);

    // Grab some fixed parameters
    ohmd_device_geti(hmdDevice, OHMD_SCREEN_HORIZONTAL_RESOLUTION, &hmd_width);
    ohmd_device_geti(hmdDevice, OHMD_SCREEN_VERTICAL_RESOLUTION, &hmd_height);
    eye_width = hmd_width / 2;
    eye_height = hmd_height;
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

  // Open the cameras
  stereoCamera = new ArgusCamera(demoState.display, demoState.context, {LEFT_CAMERA_INDEX, RIGHT_CAMERA_INDEX}, s_cameraWidth, s_cameraHeight, s_cameraFramerate);

  // Generate derived data for calibration
  initCalibration();

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);

  // Calibration mode
  {
    bool needIntrinsicCalibration = true;
    bool needStereoCalibration = true;
    // Try reading calibration data from the file
    readCalibrationData();

    // Calibrate individual cameras
    if (!haveIntrinsicCalibration()) {
      doIntrinsicCalibration();

      if (want_quit)
        goto quit;

      // Incrementally save intrinsic calibration data if it was updated this run
      saveCalibrationData(); 
    }

    if (!haveStereoCalibration()) {
      doStereoCalibration();
      saveCalibrationData();
    }

    if (want_quit)
      goto quit;

    generateCalibrationDerivedData();

    // Compute new distortion maps with the now-valid stereo calibration.
    updateCameraDistortionMap(0, true);
    updateCameraDistortionMap(1, true);

  } // Calibration mode

  // Load masks.
  {
    for (size_t cameraIdx = 0; cameraIdx < 2; ++cameraIdx) {
      cameraMask[cameraIdx] = rhi()->newTexture2D(s_cameraWidth, s_cameraHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));

      int x, y, fileChannels;
      char filename[32];
      sprintf(filename, "camera%zu_mask.png", cameraIdx);
      uint8_t* maskData = stbi_load(filename, &x, &y, &fileChannels, 1);
      if (maskData && ((x != s_cameraWidth) || (y != s_cameraHeight))) {
        printf("Mask file \"%s\" dimensions %dx%d do not match camera dimensions %zux%zu. The mask will not be applied.\n", filename, x, y, s_cameraWidth, s_cameraHeight);
        free(maskData);
        maskData = NULL;
      }

      if (!maskData) {
        printf("No usable mask data found in \"%s\" for camera %zu. A template will be created.\n", filename, cameraIdx);

        x = s_cameraWidth;
        y = s_cameraHeight;
        maskData = (uint8_t*) malloc(x * y);

        // Save a snapshot from this camera as a template.
        stereoCamera->readFrame();

        RHIRenderTarget::ptr snapRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({cameraMask[cameraIdx]}));
        rhi()->beginRenderPass(snapRT, kLoadInvalidate);
        // This pipeline flips the Y axis for OpenCV's coordinate system, which is the same as the PNG coordinate system
        rhi()->bindRenderPipeline(camGreyscalePipeline);
        rhi()->loadTexture(ksImageTex, stereoCamera->rgbTexture(cameraIdx), linearClampSampler);
        rhi()->drawNDCQuad();
        rhi()->endRenderPass(snapRT);

        rhi()->readbackTexture(cameraMask[cameraIdx], 0, kVertexElementTypeUByte1N, maskData);
        char templateFilename[32];
        sprintf(templateFilename, "camera%zu_mask_template.png", cameraIdx);
        stbi_write_png(templateFilename, s_cameraWidth, s_cameraHeight, 1, maskData, /*rowBytes=*/s_cameraWidth);

        // Fill a completely white mask for upload
        memset(maskData, 0xff, x * y);
      } else {
        printf("Loaded mask data for camera %zu\n", cameraIdx);
      }

      // Y-flip the image to convert from PNG to GL coordsys
      char* flippedMask = new char[s_cameraWidth * s_cameraHeight];
      for (size_t row = 0; row < s_cameraHeight; ++row) {
        memcpy(flippedMask + (row * s_cameraWidth), maskData + (((s_cameraHeight - 1) - row) * s_cameraWidth), s_cameraWidth);
      }

      rhi()->loadTextureData(cameraMask[cameraIdx], kVertexElementTypeUByte1N, flippedMask);

      delete[] flippedMask;

      free(maskData);
    }
  }

  {
    stereoCamera->setRepeatCapture(true);

    // Camera rendering mode
    uint64_t frameCounter = 0;
    boost::accumulators::accumulator_set<double, boost::accumulators::stats<
        boost::accumulators::tag::min,
        boost::accumulators::tag::max,
        boost::accumulators::tag::mean,
        boost::accumulators::tag::median
      > > captureLatency;

    uint64_t previousCaptureTimestamp = 0;
    boost::accumulators::accumulator_set<double, boost::accumulators::stats<
        boost::accumulators::tag::min,
        boost::accumulators::tag::max,
        boost::accumulators::tag::mean,
        boost::accumulators::tag::median
      > > captureInterval;

    uint64_t previousFrameTimestamp = 0;
    boost::accumulators::accumulator_set<double, boost::accumulators::stats<
        boost::accumulators::tag::min,
        boost::accumulators::tag::max,
        boost::accumulators::tag::mean,
        boost::accumulators::tag::median
      > > frameInterval;

    io.DeltaTime = 1.0f / 60.0f; // Will be updated during frame-timing computation

    RHIRenderTarget::ptr guiRT;
    RHISurface::ptr guiTex;

    guiTex = rhi()->newTexture2D(io.DisplaySize.x * io.DisplayFramebufferScale.x, io.DisplaySize.y * io.DisplayFramebufferScale.y, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    guiRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({ guiTex }));

    while (!want_quit) {
      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplInputListener_NewFrame();
      ImGui::NewFrame();

      ++frameCounter;

      stereoCamera->readFrame();

      if (previousCaptureTimestamp) {
        double interval_ms = static_cast<double>(stereoCamera->sensorTimestamp(0) - previousCaptureTimestamp) / 1000000.0;
        captureInterval(interval_ms);
      }
      previousCaptureTimestamp = stereoCamera->sensorTimestamp(0);


      if ((frameCounter & 0x7fUL) == 0) {
        printf("Capture latency: min=%.3g max=%.3g mean=%.3g median=%.3g\n",
          boost::accumulators::min(captureLatency),
          boost::accumulators::max(captureLatency),
          boost::accumulators::mean(captureLatency),
          boost::accumulators::median(captureLatency));

        captureLatency = {};

        printf("Capture interval: min=%.3g max=%.3g mean=%.3g median=%.3g\n",
          boost::accumulators::min(captureInterval),
          boost::accumulators::max(captureInterval),
          boost::accumulators::mean(captureInterval),
          boost::accumulators::median(captureInterval));

        captureInterval = {};

        printf("Frame interval: % .6f ms (% .6f fps) min=%.3g max=%.3g median=%.3g\n",
          static_cast<double>(boost::accumulators::mean(frameInterval)) / 1000000.0,
          1000000000.0 / static_cast<double>(boost::accumulators::mean(frameInterval)),

          static_cast<double>(boost::accumulators::min(frameInterval)) / 1000000.0,
          static_cast<double>(boost::accumulators::max(frameInterval)) / 1000000.0,
          static_cast<double>(boost::accumulators::median(frameInterval)) / 1000000.0);

        frameInterval = {};

        //printf("CLOCK_MONOTONIC: %llu. Sensor timestamps: %llu %llu\n", raw_ns, stereoCamera->sensorTimestamp(0), stereoCamera->sensorTimestamp(1));
      }


      {
        // GUI support
        ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x*0.5f, io.DisplaySize.y), 0, /*pivot=*/ImVec2(0.5f, 1.0f)); // bottom-center aligned
        ImGui::Begin("Overlay", NULL, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);
        ImGui::Text("Config");
        ImGui::SliderFloat("Scale", &scaleFactor, 0.1f, 5.0f);
        ImGui::SliderFloat("Stereo Offset", &stereoOffset, -2.0f, 2.0f);
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::End();

        rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
        rhi()->beginRenderPass(guiRT, kLoadClear);
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        rhi()->endRenderPass(guiRT);
      }

      for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
        rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        rhi()->beginRenderPass(eyeRT[eyeIndex], kLoadClear);

        rhi()->bindRenderPipeline(camUndistortMaskPipeline);
        rhi()->loadTexture(ksImageTex, stereoCamera->rgbTexture(eyeIndex), linearClampSampler);
        rhi()->loadTexture(ksDistortionMap, cameraDistortionMap[eyeIndex], linearClampSampler);
        rhi()->loadTexture(ksMaskTex, cameraMask[eyeIndex], linearClampSampler);


        // coordsys right now: -X = left, -Z = into screen
        // (camera is at the origin)
        float stereoOffsetSign = (eyeIndex == 0 ? -1.0f : 1.0f);
        const glm::vec3 tx = glm::vec3(stereoOffsetSign * stereoOffset, 0.0f, -7.0f);
        glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(scaleFactor * (static_cast<float>(stereoCamera->streamWidth()) / static_cast<float>(stereoCamera->streamHeight())), scaleFactor, 1.0f)); // TODO
        // Intentionally ignoring the eyeView matrix here. Camera to eye stereo offset is controlled directly by the stereoOffset variable
        glm::mat4 mvp = eyeProjection[eyeIndex] * model;

        float uClipFrac = 0.75f;
#if 1
        // Compute the clipping parameters to cut the views off at the centerline of the view (x=0 in model space)
        {
          glm::vec3 worldP0 = glm::vec3(model * glm::vec4(-1.0f, 0.0f, 0.0f, 1.0f));
          glm::vec3 worldP1 = glm::vec3(model * glm::vec4( 1.0f, 0.0f, 0.0f, 1.0f));

          float xLen = fabs(worldP0.x - worldP1.x);
          // the coordinate that's closest to X0 will be the one we want to clip
          float xOver = std::min<float>(fabs(worldP0.x), fabs(worldP1.x));

          uClipFrac = (xLen - xOver)/xLen;
        }
#endif

        NDCClippedQuadUniformBlock ub;
        ub.modelViewProjection = mvp;
        if (eyeIndex == 0) { // left
          ub.minUV = glm::vec2(0.0f,  0.0f);
          ub.maxUV = glm::vec2(uClipFrac, 1.0f);
        } else { // right
          ub.minUV = glm::vec2(1.0f - uClipFrac, 0.0f);
          ub.maxUV = glm::vec2(1.0f,  1.0f);
        }

        rhi()->loadUniformBlockImmediate(ksNDCClippedQuadUniformBlock, &ub, sizeof(NDCClippedQuadUniformBlock));

        rhi()->drawNDCQuad();

        // UI overlay
        rhi()->bindBlendState(standardAlphaOverBlendState);
        rhi()->bindRenderPipeline(uiLayerPipeline);
        rhi()->loadTexture(ksImageTex, guiTex, linearClampSampler);
        UILayerUniformBlock uiLayerBlock;
        uiLayerBlock.modelViewProjection = eyeProjection[eyeIndex] * eyeView[eyeIndex] * glm::translate(glm::vec3(0.0f, 0.0f, -2.0f)) * glm::scale(glm::vec3(io.DisplaySize.x / io.DisplaySize.y, 1.0f, 1.0f));

        rhi()->loadUniformBlockImmediate(ksUILayerUniformBlock, &uiLayerBlock, sizeof(UILayerUniformBlock));
        rhi()->drawNDCQuad();

        rhi()->endRenderPass(eyeRT[eyeIndex]);
      }

      renderHMDFrame();
      {
        uint64_t thisFrameTimestamp = currentTimeNs();
        if (previousFrameTimestamp) {
          uint64_t interval = thisFrameTimestamp - previousFrameTimestamp;
          //if ((frameCounter & 0xff)  == 0xff) {
          //  printf("raw interval %lu\n", interval);
          //}
          frameInterval(interval);

          // Update the target capture interval periodically
#if 0
          if ((frameCounter & 0x1f) == 0x1f) {
            stereoCamera->setTargetCaptureIntervalNs(boost::accumulators::rolling_mean(frameInterval));
          }
#endif

          io.DeltaTime = static_cast<double>(interval / 1000000000.0);
        }

        captureLatency(static_cast<double>(thisFrameTimestamp - stereoCamera->sensorTimestamp(0)) / 1000000.0);

        previousFrameTimestamp = thisFrameTimestamp;
      }
    } // Camera rendering loop
  }
quit:
  // Restore signal handlers
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);

  // clear screen
  rhi()->beginRenderPass(windowRenderTarget, kLoadClear);
  rhi()->endRenderPass(windowRenderTarget);
  rhi()->swapBuffers(windowRenderTarget);

  stereoCamera->stop();
  delete stereoCamera;

  // Release OpenGL resources
  eglMakeCurrent( demoState.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT );
  eglDestroySurface( demoState.display, demoState.surface );
  eglDestroyContext( demoState.display, demoState.context );
  eglTerminate( demoState.display );

  return 0;
}

