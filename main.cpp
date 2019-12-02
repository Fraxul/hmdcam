#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>
#include <vector>

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

#include "openhmd/openhmd.h"

//#define SAVE_CALIBRATION_IMAGES
#ifdef SAVE_CALIBRATION_IMAGES
  #define STB_IMAGE_WRITE_IMPLEMENTATION
  #include "../stb/stb_image_write.h"
#endif

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

// Camera config
#define SWAP_CAMERA_EYES
#define CAMERA_INVERTED 1 // 0 = upright, 1 = camera rotated 180 degrees. (90 degree rotation is not supported)


// Mapping of libargus camera device index to camera[0] (left) and camera[1] (right).
#ifdef SWAP_CAMERA_EYES
  #define LEFT_CAMERA_INDEX 1
  #define RIGHT_CAMERA_INDEX 0
#else
  #define LEFT_CAMERA_INDEX 0
  #define RIGHT_CAMERA_INDEX 1
#endif

static cv::Size calibrationBoardSize(9, 6);


RHIRenderTarget::ptr windowRenderTarget;

struct NDCQuadUniformBlock {
  glm::mat4 modelViewProjection;
};
FxAtomicString ksNDCQuadUniformBlock("NDCQuadUniformBlock");
RHIRenderPipeline::ptr camTexturedQuadPipeline;
RHIRenderPipeline::ptr camOverlayPipeline;
RHIRenderPipeline::ptr camOverlayStereoPipeline;
RHIRenderPipeline::ptr camInvDistortionPipeline;
RHIRenderPipeline::ptr camGreyscalePipeline;
RHIRenderPipeline::ptr camGreyscaleUndistortPipeline;

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
FxAtomicString ksLeftCameraTex("leftCameraTex");
FxAtomicString ksRightCameraTex("rightCameraTex");
FxAtomicString ksLeftDistortionMap("leftDistortionMap");
FxAtomicString ksRightDistortionMap("rightDistortionMap");
FxAtomicString ksOverlayTex("overlayTex");
FxAtomicString ksDistortionMap("distortionMap");

// per-eye render targets (pre distortion)
RHISurface::ptr eyeTex[2];
RHIRenderTarget::ptr eyeRT[2];
// per-eye distortion parameter buffers
RHIBuffer::ptr hmdDistortionParams[2];

// HMD info/state
ohmd_context* hmdContext = NULL;
ohmd_device* hmdDevice = NULL;
int hmd_width, hmd_height;
int eye_width, eye_height;
bool rotate_screen = false;
glm::mat4 eyeProjection[2];

// Camera info/state
ArgusCamera* camera[2];
RHISurface::ptr cameraDistortionMap[2];
cv::Mat cameraMatrix[2];
cv::Mat distCoeffs[2];

cv::Mat stereoRotation, stereoTranslation; // Calibrated
cv::Mat stereoRectification[2], stereoProjection[2]; // Derived from stereoRotation/stereoTranslation via cv::stereoRectify
cv::Mat stereoDisparityToDepth;


uint64_t currentTimeUs() {
  return boost::chrono::duration_cast<boost::chrono::microseconds>(boost::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t currentTimeMs() {
  return currentTimeUs() / 1000ULL;
}

void updateCameraDistortionMap(size_t cameraIdx, bool useStereoCalibration)  {
  cv::Size imageSize = cv::Size(1280, 720);
  float alpha = 0.25; // scaling factor. 0 = no invalid pixels in output (no black borders), 1 = use all input pixels
  cv::Mat map1, map2;
  if (useStereoCalibration) {
    cv::initUndistortRectifyMap(cameraMatrix[cameraIdx], distCoeffs[cameraIdx], stereoRectification[cameraIdx], stereoProjection[cameraIdx], imageSize, CV_32F, map1, map2);
  } else {
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix[cameraIdx], distCoeffs[cameraIdx], imageSize, alpha, cv::Size(), NULL, /*centerPrincipalPoint=*/true);
    cv::initUndistortRectifyMap(cameraMatrix[cameraIdx], distCoeffs[cameraIdx], cv::noArray(), newCameraMatrix, imageSize, CV_32F, map1, map2);
  }
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

  cameraDistortionMap[cameraIdx] = rhi()->newTexture2D(imageSize.width, imageSize.height, RHISurfaceDescriptor(kSurfaceFormat_RG32f));

  rhi()->loadTextureData(cameraDistortionMap[cameraIdx], kVertexElementTypeFloat2, distortionMapTmp);

  delete[] distortionMapTmp;
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
    "shaders/camInvDistortion.frag.glsl",
    ndcQuadVertexLayout);
    desc.setFlag("CAMERA_INVERTED", (bool) CAMERA_INVERTED);
    camInvDistortionPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
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

  ohmd_device_getf(hmdDevice, OHMD_EYE_IPD, &ipd);
  //viewport is half the screen
  ohmd_device_getf(hmdDevice, OHMD_SCREEN_HORIZONTAL_SIZE, &(viewport_scale[0]));
  viewport_scale[0] /= 2.0f;
  ohmd_device_getf(hmdDevice, OHMD_SCREEN_VERTICAL_SIZE, &(viewport_scale[1]));
  //distortion coefficients
  ohmd_device_getf(hmdDevice, OHMD_UNIVERSAL_DISTORTION_K, &(distortion_coeffs[0]));
  ohmd_device_getf(hmdDevice, OHMD_UNIVERSAL_ABERRATION_K, &(aberr_scale[0]));
  //calculate lens centers (assuming the eye separation is the distance between the lens centers)
  ohmd_device_getf(hmdDevice, OHMD_LENS_HORIZONTAL_SEPARATION, &sep);
  ohmd_device_getf(hmdDevice, OHMD_LENS_VERTICAL_POSITION, &(left_lens_center[1]));
  ohmd_device_getf(hmdDevice, OHMD_LENS_VERTICAL_POSITION, &(right_lens_center[1]));
  left_lens_center[0] = viewport_scale[0] - sep/2.0f;
  right_lens_center[0] = sep/2.0f;
  //assume calibration was for lens view to which ever edge of screen is further away from lens center
  warp_scale = (left_lens_center[0] > right_lens_center[0]) ? left_lens_center[0] : right_lens_center[0];
  warp_adj = 1.0f;

  // Setup projection matrices
  ohmd_device_getf(hmdDevice, OHMD_LEFT_EYE_GL_PROJECTION_MATRIX, &(eyeProjection[0][0][0]));
  ohmd_device_getf(hmdDevice, OHMD_RIGHT_EYE_GL_PROJECTION_MATRIX, &(eyeProjection[1][0][0]));
  // Cook the stereo separation transform into the projection matrices
  // TODO stereo separation scale
  eyeProjection[0] = eyeProjection[0] * glm::translate(glm::vec3(ipd *  10.0f, 0.0f, 0.0f));
  eyeProjection[1] = eyeProjection[1] * glm::translate(glm::vec3(ipd * -10.0f, 0.0f, 0.0f));

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

    rhi()->bindRenderPipeline(hmdDistortionPipeline);
    rhi()->loadUniformBlock(ksHMDDistortionUniformBlock, hmdDistortionParams[eyeIndex]);
    rhi()->loadTexture(ksImageTex, eyeTex[eyeIndex]);

    rhi()->drawNDCQuad();
  }

  rhi()->endRenderPass(windowRenderTarget);

  rhi()->swapBuffers(windowRenderTarget);
}

cv::Mat captureGreyscale(size_t cameraIdx, RHISurface::ptr tex, RHIRenderTarget::ptr rt) {

  rhi()->beginRenderPass(rt, kLoadInvalidate);
  rhi()->bindRenderPipeline(camGreyscalePipeline);
  rhi()->loadTexture(ksImageTex, camera[cameraIdx]->rgbTexture());
  rhi()->drawNDCQuad();
  rhi()->endRenderPass(rt);

  cv::Mat res;
  res.create(/*rows=*/ tex->height(), /*columns=*/tex->width(), CV_8UC1);
  assert(res.isContinuous());
  rhi()->readbackTexture(tex, 0, kVertexElementTypeUByte1N, res.ptr(0));
  return res;
}

cv::Mat captureSBSUndistortedStereoGreyscale(RHISurface::ptr tex, RHIRenderTarget::ptr rt) {

  rhi()->beginRenderPass(rt, kLoadInvalidate);
  rhi()->bindRenderPipeline(camGreyscaleUndistortPipeline);

  rhi()->setViewport(RHIRect::xywh(0, 0, tex->width() / 2, tex->height()));
  rhi()->loadTexture(ksImageTex, camera[0]->rgbTexture());
  rhi()->loadTexture(ksDistortionMap, cameraDistortionMap[0]);
  rhi()->drawNDCQuad();

  rhi()->setViewport(RHIRect::xywh(tex->width() / 2, 0, tex->width() / 2, tex->height()));
  rhi()->loadTexture(ksImageTex, camera[1]->rgbTexture());
  rhi()->loadTexture(ksDistortionMap, cameraDistortionMap[1]);
  rhi()->drawNDCQuad();
  rhi()->endRenderPass(rt);

  cv::Mat res;
  res.create(/*rows=*/ tex->height(), /*columns=*/tex->width(), CV_8UC1);
  assert(res.isContinuous());
  rhi()->readbackTexture(tex, 0, kVertexElementTypeUByte1N, res.ptr(0));
  return res;
}

static double computeReprojectionErrors(
        const std::vector<std::vector<cv::Point3f> >& objectPoints,
        const std::vector<std::vector<cv::Point2f> >& imagePoints,
        const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs,
        const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs,
        std::vector<float>& perViewErrors )
{
    std::vector<cv::Point2f> imagePoints2;
    int i, totalPoints = 0;
    double totalErr = 0, err;
    perViewErrors.resize(objectPoints.size());

    for( i = 0; i < (int)objectPoints.size(); i++ )
    {
        cv::projectPoints(cv::Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix, distCoeffs, imagePoints2);
        err = cv::norm(cv::Mat(imagePoints[i]), cv::Mat(imagePoints2), cv::NORM_L2);
        int n = (int)objectPoints[i].size();
        perViewErrors[i] = (float)std::sqrt(err*err/n);
        totalErr += err*err;
        totalPoints += n;
    }

    return std::sqrt(totalErr/totalPoints);
}

void drawStatusLines(cv::Mat& image, const std::vector<std::string> lines) {
  std::vector<cv::Size> lineSizes; // total size of bounding rect per line
  std::vector<int> baselines; // Y size of area above baseline
  std::vector<int> lineYOffsets; // computed Y coordinate of line in drawing stack

  uint32_t rectPadding = 4;
  uint32_t linePaddingY = 4;

  for (size_t lineIdx = 0; lineIdx < lines.size(); ++lineIdx) {
    int baseline = 0;
    lineSizes.push_back(cv::getTextSize(lines[lineIdx].c_str(), 1, 1, 1, &baseline));
    baselines.push_back(baseline);
    //printf("line [%u] \"%s\" size: %u x %u baseline: %u\n", lineIdx, lines[lineIdx].c_str(), lineSizes[lineIdx].width, lineSizes[lineIdx].height, baselines[lineIdx]);
  }

  // Compute overall size of center-justified text line stack
  cv::Size boundSize;
  for (size_t lineIdx = 0; lineIdx < lines.size(); ++lineIdx) {
    boundSize.width = std::max(boundSize.width, lineSizes[lineIdx].width);
    lineYOffsets.push_back(boundSize.height);
    boundSize.height += baselines[lineIdx]; // only counting area above the baseline, descenders can overlap the subsequent line
    if (lineIdx == (lines.size() - 1)) {
      // add the area under the baseline back in for the last line, since there are no subsequent lines for it to overlap
      boundSize.height += lineSizes[lineIdx].height -  baselines[lineIdx];
    } else {
      // add inter-line padding
      boundSize.height += linePaddingY;
    }
  }

  cv::Point origin; // left-top of drawing region
  origin.x = (image.cols / 2) - (boundSize.width / 2);
  origin.y = image.rows - (boundSize.height + rectPadding);

  // Draw background rect
  cv::rectangle(image,
    cv::Point(origin.x - rectPadding, origin.y - rectPadding),
    cv::Point(origin.x + boundSize.width + rectPadding, origin.y + boundSize.height + rectPadding),
    cv::Scalar(1, 1, 1), CV_FILLED);

  // Draw lines
  for (size_t lineIdx = 0; lineIdx < lines.size(); ++lineIdx) {
    cv::putText(image, lines[lineIdx].c_str(),
      cv::Point(
        origin.x + ((boundSize.width - lineSizes[lineIdx].width) / 2),
        origin.y + lineYOffsets[lineIdx] + lineSizes[lineIdx].height),
      1, 1, cv::Scalar(0, 255, 0));
  }
}

int main(int argc, char* argv[]) {

  init_ogl();

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

  const size_t cameraWidth = 1280, cameraHeight = 720;


  camera[0] = new ArgusCamera(demoState.display, demoState.context, LEFT_CAMERA_INDEX, cameraWidth, cameraHeight);
  camera[1] = new ArgusCamera(demoState.display, demoState.context, RIGHT_CAMERA_INDEX, cameraWidth, cameraHeight);

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);

  // Calibration mode
  {
    bool needIntrinsicCalibration = true;
    bool needStereoCalibration = true;
    // Try reading calibration data from the file
    {
      cv::FileStorage fs("calibration.yml", cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
      if (fs.isOpened()) {
        try {
          fs["camera0_matrix"] >> cameraMatrix[0];
          fs["camera1_matrix"] >> cameraMatrix[1];
          fs["camera0_distortionCoeffs"] >> distCoeffs[0];
          fs["camera1_distortionCoeffs"] >> distCoeffs[1];

          if (!(cameraMatrix[0].empty() || cameraMatrix[1].empty() || distCoeffs[0].empty() || distCoeffs[1].empty())) {
            printf("Loaded camera intrinsic calibration data from file\n");
            // Build initial intrinsic-only distortion maps
            updateCameraDistortionMap(0, false);
            updateCameraDistortionMap(1, false);
            needIntrinsicCalibration = false;
          }

          fs["stereoRotation"] >> stereoRotation;
          fs["stereoTranslation"] >> stereoTranslation;
          if (!(stereoRotation.empty() || stereoTranslation.empty())) {
            needStereoCalibration = false;
            printf("Loaded stereo offset calibration data from file\n");
          }

        } catch (const std::exception& ex) {
          printf("Unable to read calibration data: %s\n", ex.what());
        }

      } else {
        printf("Unable to open calibration data file\n");
      }
    }

    // Calibrate individual cameras
    if (needIntrinsicCalibration) {
      // Textures and RTs we use for half and full-res captures
      RHISurface::ptr halfGreyTex = rhi()->newTexture2D(cameraWidth / 2, cameraHeight / 2, RHISurfaceDescriptor(kSurfaceFormat_R8));
      RHIRenderTarget::ptr halfGreyRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({halfGreyTex}));

      RHISurface::ptr fullGreyTex = rhi()->newTexture2D(cameraWidth, cameraHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));
      RHIRenderTarget::ptr fullGreyRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({fullGreyTex}));

      RHISurface::ptr feedbackTex = rhi()->newTexture2D(cameraWidth / 2, cameraHeight / 2, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));

      cv::Mat feedbackHalfResView;
      feedbackHalfResView.create(/*rows=*/ cameraHeight / 2, /*columns=*/cameraWidth / 2, CV_8UC4);

      for (unsigned int cameraIdx = 0; cameraIdx < 2; ++cameraIdx) {
        printf("Camera %u intrinsic calibration\n", cameraIdx);
        const unsigned int targetSampleCount = 10;
        const uint64_t captureDelayMs = 500;
        uint64_t lastCaptureTime = 0;

        std::vector<std::vector<cv::Point2f> > calibrationPoints;

        while (!want_quit && calibrationPoints.size() < targetSampleCount) {
          camera[cameraIdx]->readFrame();
          cv::Mat viewHalfRes = captureGreyscale(cameraIdx, halfGreyTex, halfGreyRT);
          cv::Mat viewFullRes = captureGreyscale(cameraIdx, fullGreyTex, fullGreyRT);

          std::vector<cv::Point2f> halfResPoints;

          // Run initial search on the half-res image for speed
          //bool found = cv::findChessboardCorners(viewHalfRes, calibrationBoardSize, halfResPoints, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);
          bool found = cv::findChessboardCorners(viewHalfRes, calibrationBoardSize, halfResPoints, cv::CALIB_CB_FAST_CHECK);

          std::vector<cv::Point2f> fullResPoints;
          if (found) {
            // Map points from the half-res image to the full-res image
            for (size_t i = 0; i < halfResPoints.size(); ++i) {
              fullResPoints.push_back(halfResPoints[i] * 2.0f);
            }

            // Improve accuracy by running the subpixel corner search on the full-res view
            cv::cornerSubPix(viewFullRes, fullResPoints, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));

            // Map points back to half-res for the overlay
            halfResPoints.clear();
            for (size_t i = 0; i < fullResPoints.size(); ++i) {
              halfResPoints.push_back(fullResPoints[i] * 0.5f);
            }
          }

          // Draw feedback points
          memset(feedbackHalfResView.ptr(0), 0, feedbackHalfResView.total() * 4);
          cv::drawChessboardCorners( feedbackHalfResView, calibrationBoardSize, cv::Mat(halfResPoints), found );

          char status1[64];
          char status2[64];
          sprintf(status1, "Camera %u (%s)", cameraIdx, cameraIdx == 0 ? "left" : "right");
          sprintf(status2, "%zu/%u samples", calibrationPoints.size(), targetSampleCount);

          drawStatusLines(feedbackHalfResView, { status1, "Intrinsic calibration", status2 } );

          rhi()->loadTextureData(feedbackTex, kVertexElementTypeUByte4N, feedbackHalfResView.ptr(0));

          if (found && currentTimeMs() > (lastCaptureTime + captureDelayMs)) {

  #ifdef SAVE_CALIBRATION_IMAGES
            char filename1[64];
            char filename2[64];
            static int fileIdx = 0;
            ++fileIdx;
            sprintf(filename1, "calib_%u_%02u_frame.png", cameraIdx, fileIdx);
            sprintf(filename2, "calib_%u_%02u_overlay.png", cameraIdx, fileIdx);

            stbi_write_png(filename1, cameraWidth/2, cameraHeight/2, 1, viewHalfRes.ptr(0), /*rowBytes=*/(cameraWidth/2));

            // composite with the greyscale view and fix the alpha channel before writing
            for (size_t pixelIdx = 0; pixelIdx < ((cameraWidth/2) * (cameraHeight/2)); ++pixelIdx) {
              uint8_t* p = feedbackHalfResView.ptr(0) + (pixelIdx * 4);
              if (!(p[0] || p[1] || p[2])) {
                p[0] = p[1] = p[2] = viewHalfRes.ptr(0)[pixelIdx];

              }
              p[3] = 0xff;
            }
            stbi_write_png(filename2, cameraWidth/2, cameraHeight/2, 4, feedbackHalfResView.ptr(0), /*rowBytes=*/(cameraWidth/2) * 4);
            printf("Saved %s and %s\n", filename1, filename2);
  #endif

            calibrationPoints.push_back(fullResPoints);
            lastCaptureTime = currentTimeMs();
          }

          // Draw camera stream and feedback overlay to both eye RTs

          for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
            rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            rhi()->beginRenderPass(eyeRT[eyeIndex], kLoadClear);

            rhi()->bindRenderPipeline(camOverlayPipeline);
            rhi()->loadTexture(ksImageTex, camera[cameraIdx]->rgbTexture());
            rhi()->loadTexture(ksOverlayTex, feedbackTex);

            // coordsys right now: -X = left, -Z = into screen
            // (camera is at the origin)
            const glm::vec3 tx = glm::vec3(0.0f, 0.0f, -7.0f);
            const float scaleFactor = 5.0f;
            glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(scaleFactor * (static_cast<float>(cameraWidth) / static_cast<float>(cameraHeight)), scaleFactor, 1.0f)); // TODO
            glm::mat4 mvp = eyeProjection[eyeIndex] * model;

            NDCQuadUniformBlock ub;
            ub.modelViewProjection = mvp;
            rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(NDCQuadUniformBlock));

            rhi()->drawNDCQuad();

            rhi()->endRenderPass(eyeRT[eyeIndex]);
          }
          renderHMDFrame();

        }

        if (want_quit)
          break;

        // Calibration samples collected
        {

          std::vector<cv::Mat> rvecs, tvecs;
          std::vector<float> reprojErrs;
          cv::Size imageSize(cameraWidth, cameraHeight);
          float squareSize = 1.0f;
          float aspectRatio = 1.0f;
          int flags = cv::CALIB_FIX_ASPECT_RATIO;

          cameraMatrix[cameraIdx] = cv::Mat::eye(3, 3, CV_64F);
          if( flags & cv::CALIB_FIX_ASPECT_RATIO )
              cameraMatrix[cameraIdx].at<double>(0,0) = aspectRatio;
          distCoeffs[cameraIdx] = cv::Mat::zeros(8, 1, CV_64F);

          std::vector<std::vector<cv::Point3f> > objectPoints(1);

          for (int i = 0; i < calibrationBoardSize.height; ++i) {
              for (int j = 0; j < calibrationBoardSize.width; ++j) {
                  objectPoints[0].push_back(cv::Point3f(static_cast<float>(j)*squareSize, static_cast<float>(i)*squareSize, 0));
              }
          }

          objectPoints.resize(calibrationPoints.size(), objectPoints[0]);

          double rms = cv::calibrateCamera(objectPoints, calibrationPoints, imageSize, cameraMatrix[cameraIdx],
                          distCoeffs[cameraIdx], rvecs, tvecs, flags|cv::CALIB_FIX_K4|cv::CALIB_FIX_K5);
          printf("RMS error reported by calibrateCamera: %g\n", rms);

          bool ok = cv::checkRange(cameraMatrix[cameraIdx]) && cv::checkRange(distCoeffs[cameraIdx]);

          double totalAvgErr = computeReprojectionErrors(objectPoints, calibrationPoints, rvecs, tvecs, cameraMatrix[cameraIdx], distCoeffs[cameraIdx], reprojErrs);

          printf("%s. avg reprojection error = %.2f\n",
                 ok ? "Calibration succeeded" : "Calibration failed",
                 totalAvgErr);

          // Build initial intrinsic-only distortion map
          updateCameraDistortionMap(cameraIdx, false);
        }
      } // Per-camera calibration loop

    } // Individual camera calibration

    if (needStereoCalibration) {
      // Stereo pair calibration

      // Textures and RTs we use for full-res captures.
      // We skip half-res captures for the full stereo calibration -- the target will be farther from the cameras,
      // so we will probably need the extra resolution to find it.
      RHISurface::ptr fullGreyTex = rhi()->newTexture2D(cameraWidth * 2, cameraHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));
      RHIRenderTarget::ptr fullGreyRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({fullGreyTex}));

      RHISurface::ptr feedbackTex = rhi()->newTexture2D(cameraWidth, cameraHeight / 2, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));

      cv::Mat feedbackHalfResViewStereo;
      feedbackHalfResViewStereo.create(/*rows=*/ cameraHeight / 2, /*columns=*/cameraWidth, CV_8UC4);

      const unsigned int targetSampleCount = 10;
      const uint64_t captureDelayMs = 500;
      uint64_t lastCaptureTime = 0;

      std::vector<std::vector<cv::Point2f> > calibrationPoints[2];

      while (!want_quit && calibrationPoints[0].size() < targetSampleCount) {

        camera[0]->readFrame();
        camera[1]->readFrame();

        cv::Mat viewFullResStereo = captureSBSUndistortedStereoGreyscale(fullGreyTex, fullGreyRT);

        // Create monoscopic views of SbS-stereo images
        cv::Mat viewFullRes[2];
        cv::Mat feedbackHalfResView[2];
        viewFullRes[0] = viewFullResStereo.colRange(0, viewFullResStereo.cols / 2);
        viewFullRes[1] = viewFullResStereo.colRange(viewFullResStereo.cols / 2, viewFullResStereo.cols);
        feedbackHalfResView[0] = feedbackHalfResViewStereo.colRange(0, feedbackHalfResViewStereo.cols / 2);
        feedbackHalfResView[1] = feedbackHalfResViewStereo.colRange(feedbackHalfResViewStereo.cols / 2, feedbackHalfResViewStereo.cols);


        std::vector<cv::Point2f> fullResPoints[2];
        // Run initial search
        bool foundLeft = cv::findChessboardCorners(viewFullRes[0], calibrationBoardSize, fullResPoints[0], cv::CALIB_CB_FAST_CHECK);
        bool foundRight =  cv::findChessboardCorners(viewFullRes[1], calibrationBoardSize, fullResPoints[1], cv::CALIB_CB_FAST_CHECK);
        bool found = foundLeft && foundRight;

        if (found) {
          // Improve accuracy by running the subpixel corner search
          cv::cornerSubPix(viewFullRes[0], fullResPoints[0], cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
          cv::cornerSubPix(viewFullRes[1], fullResPoints[1], cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
        }

        std::vector<cv::Point2f> halfResPoints[2];
        // Map points back to half-res for the overlay
        for (size_t viewIdx = 0; viewIdx < 2; ++viewIdx) {
          for (size_t i = 0; i < fullResPoints[viewIdx].size(); ++i) {
            halfResPoints[viewIdx].push_back(fullResPoints[viewIdx][i] * 0.5f);
          }
        }

        // Draw feedback points
        memset(feedbackHalfResViewStereo.ptr(0), 0, feedbackHalfResViewStereo.total() * 4);
        cv::drawChessboardCorners(feedbackHalfResView[0], calibrationBoardSize, cv::Mat(halfResPoints[0]), foundLeft);
        cv::drawChessboardCorners(feedbackHalfResView[1], calibrationBoardSize, cv::Mat(halfResPoints[1]), foundRight);

        char status1[64];
        sprintf(status1, "%zu/%u samples", calibrationPoints[0].size(), targetSampleCount);

        drawStatusLines(feedbackHalfResViewStereo, { "Stereo calibration", status1 } );

        rhi()->loadTextureData(feedbackTex, kVertexElementTypeUByte4N, feedbackHalfResViewStereo.ptr(0));

        if (found && currentTimeMs() > (lastCaptureTime + captureDelayMs)) {

#if 0 //def SAVE_CALIBRATION_IMAGES
          char filename1[64];
          char filename2[64];
          static int fileIdx = 0;
          ++fileIdx;
          sprintf(filename1, "calib_stereo_%02u_frame.png", fileIdx);
          sprintf(filename2, "calib_stereo_%02u_overlay.png", fileIdx);

          stbi_write_png(filename1, cameraWidth, cameraHeight/2, 1, viewHalfRes.ptr(0), /*rowBytes=*/(cameraWidth));

          // composite with the greyscale view and fix the alpha channel before writing
          for (size_t pixelIdx = 0; pixelIdx < (cameraWidth * (cameraHeight/2)); ++pixelIdx) {
            uint8_t* p = feedbackHalfResView.ptr(0) + (pixelIdx * 4);
            if (!(p[0] || p[1] || p[2])) {
              p[0] = p[1] = p[2] = viewHalfRes.ptr(0)[pixelIdx];

            }
            p[3] = 0xff;
          }
          stbi_write_png(filename2, cameraWidth, cameraHeight/2, 4, feedbackHalfResView.ptr(0), /*rowBytes=*/cameraWidth * 4);
          printf("Saved %s and %s\n", filename1, filename2);
#endif

          calibrationPoints[0].push_back(fullResPoints[0]);
          calibrationPoints[1].push_back(fullResPoints[1]);
          lastCaptureTime = currentTimeMs();
        }

        // Draw camera stream and feedback overlay to both eye RTs

        for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
          rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          rhi()->beginRenderPass(eyeRT[eyeIndex], kLoadClear);

          rhi()->bindRenderPipeline(camOverlayStereoPipeline);
          rhi()->loadTexture(ksLeftCameraTex, camera[0]->rgbTexture());
          rhi()->loadTexture(ksLeftDistortionMap, cameraDistortionMap[0]);
          rhi()->loadTexture(ksRightCameraTex, camera[1]->rgbTexture());
          rhi()->loadTexture(ksRightDistortionMap, cameraDistortionMap[1]);
          rhi()->loadTexture(ksOverlayTex, feedbackTex);

          // coordsys right now: -X = left, -Z = into screen
          // (camera is at the origin)
          const glm::vec3 tx = glm::vec3(0.0f, 0.0f, -7.0f);
          const float scaleFactor = 2.5f;
          glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(scaleFactor * (static_cast<float>(cameraWidth*2) / static_cast<float>(cameraHeight)), scaleFactor, 1.0f)); // TODO
          glm::mat4 mvp = eyeProjection[eyeIndex] * model;

          NDCQuadUniformBlock ub;
          ub.modelViewProjection = mvp;
          rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(NDCQuadUniformBlock));

          rhi()->drawNDCQuad();

          rhi()->endRenderPass(eyeRT[eyeIndex]);
        }
        renderHMDFrame();

      } // Stereo calibration sample-gathering loop

      // Samples collected, run calibration
      std::vector<std::vector<cv::Point3f> > objectPoints(1);
      float squareSize = 1.0f;

      for (int i = 0; i < calibrationBoardSize.height; ++i) {
          for (int j = 0; j < calibrationBoardSize.width; ++j) {
              objectPoints[0].push_back(cv::Point3f(static_cast<float>(j)*squareSize, static_cast<float>(i)*squareSize, 0));
          }
      }

      objectPoints.resize(calibrationPoints[0].size(), objectPoints[0]);

      cv::Mat E, F;

      cv::stereoCalibrate(objectPoints,
        calibrationPoints[0], calibrationPoints[1],
        cameraMatrix[0], distCoeffs[0],
        cameraMatrix[1], distCoeffs[1],
        cv::Size(cameraWidth, cameraHeight),
        stereoRotation, stereoTranslation, E, F, cv::CALIB_FIX_INTRINSIC | cv::CALIB_SAME_FOCAL_LENGTH);

    } // needStereoCalibration

    // Save calibration data if it was updated this run
    if (needStereoCalibration || needIntrinsicCalibration) {
      cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
      fs.write("camera0_matrix", cameraMatrix[0]);
      fs.write("camera1_matrix", cameraMatrix[1]);
      fs.write("camera0_distortionCoeffs", distCoeffs[0]);
      fs.write("camera1_distortionCoeffs", distCoeffs[1]);
      fs.write("stereoRotation", stereoRotation);
      fs.write("stereoTranslation", stereoTranslation);
      printf("Saved updated calibration data\n");
    }

    // Compute rectification/projection transforms from the stereo calibration data
    float alpha = 0.25;
    cv::stereoRectify(
      cameraMatrix[0], distCoeffs[0],
      cameraMatrix[1], distCoeffs[1],
      cv::Size(cameraWidth, cameraHeight),
      stereoRotation, stereoTranslation,
      stereoRectification[0], stereoRectification[1],
      stereoProjection[0], stereoProjection[1],
      stereoDisparityToDepth,
      /*flags=*/0, alpha);

    // Compute new distortion maps with the now-valid stereo calibration
    updateCameraDistortionMap(0, true);
    updateCameraDistortionMap(1, true);


  } // Calibration mode


  while (!want_quit) {
    // Camera rendering mode
    {
      camera[0]->readFrame();
      camera[1]->readFrame();


      for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
        ArgusCamera* activeCamera = camera[eyeIndex];

        rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        rhi()->beginRenderPass(eyeRT[eyeIndex], kLoadClear);

        rhi()->bindRenderPipeline(camInvDistortionPipeline);
        rhi()->loadTexture(ksImageTex, activeCamera->rgbTexture());
        rhi()->loadTexture(ksDistortionMap, cameraDistortionMap[eyeIndex]);


        // coordsys right now: -X = left, -Z = into screen
        // (camera is at the origin)
        const glm::vec3 tx = glm::vec3(0.0f, 0.0f, -7.0f);
        const float scaleFactor = 5.0f;
        glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(scaleFactor * (static_cast<float>(activeCamera->streamWidth()) / static_cast<float>(activeCamera->streamHeight())), scaleFactor, 1.0f)); // TODO
        glm::mat4 mvp = eyeProjection[eyeIndex] * model;

        NDCQuadUniformBlock ub;
        ub.modelViewProjection = mvp;
        rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(NDCQuadUniformBlock));

        rhi()->drawNDCQuad();

        rhi()->endRenderPass(eyeRT[eyeIndex]);
      }

      renderHMDFrame();
      update_fps();
    }
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
  camera[1]->stop();
  delete camera[0];
  delete camera[1];

  // Release OpenGL resources
  eglMakeCurrent( demoState.display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT );
  eglDestroySurface( demoState.display, demoState.surface );
  eglDestroyContext( demoState.display, demoState.context );
  eglTerminate( demoState.display );

  return 0;
}

