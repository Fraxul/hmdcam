#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <signal.h>
#include <iostream>
#include <set>
#include <vector>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/min.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/median.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
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

#include "openhmd/openhmd.h"

// #define SAVE_CALIBRATION_IMAGES

#include <zlib.h>
static unsigned char* compress_for_stbiw(unsigned char *data, int data_len, int *out_len, int quality) {
  uLongf bufSize = compressBound(data_len);
  // note that buf will be free'd by stb_image_write.h with STBIW_FREE() (plain free() by default)
  unsigned char* buf = (unsigned char*) malloc(bufSize);
  if(buf == NULL)  return NULL;
  if(compress2(buf, &bufSize, data, data_len, quality) != Z_OK) {
    free(buf);
    return NULL;
  }
  *out_len = bufSize;
  return buf;
}
#define STBIW_ZLIB_COMPRESS compress_for_stbiw
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>

// Camera config
// Size parameters for sensor mode selection.
// Note that changing the sensor mode will invalidate the calibration
// (Pixel coordinates are baked into the calibration data)
const size_t s_cameraWidth = 1280, s_cameraHeight = 720;
//const size_t s_cameraWidth = 1920, s_cameraHeight = 1080;

// Requested capture rate for the camera. This should be the framerate of the display device, with as much precision as possible.
// TODO: autodetect this. (current value pulled from running `fbset`)
const double s_cameraFramerate = 89.527;

// #define SWAP_CAMERA_EYES
#define CAMERA_INVERTED 1 // 0 = upright, 1 = camera rotated 180 degrees. (90 degree rotation is not supported)

// ChArUco target pattern config
const cv::aruco::PREDEFINED_DICTIONARY_NAME s_charucoDictionaryName = cv::aruco::DICT_4X4_250;
const unsigned int s_charucoBoardSquareCountX = 12;
const unsigned int s_charucoBoardSquareCountY = 9;
const float s_charucoBoardSquareSideLengthMeters = 0.020f;
const float s_charucoBoardMarkerSideLengthMeters = 0.015f;

// Mapping of libargus camera device ID to index 0 (left) and 1 (right).
#ifdef SWAP_CAMERA_EYES
  #define LEFT_CAMERA_INDEX 1
  #define RIGHT_CAMERA_INDEX 0
#else
  #define LEFT_CAMERA_INDEX 0
  #define RIGHT_CAMERA_INDEX 1
#endif

cv::Ptr<cv::aruco::Dictionary> s_charucoDictionary;
cv::Ptr<cv::aruco::CharucoBoard> s_charucoBoard;

static cv::Size calibrationBoardSize_old(9, 6);


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

float scaleFactor = 2.1f;
float stereoSeparationScale = 1.0f;

// Camera info/state
ArgusCamera* stereoCamera;
RHISurface::ptr cameraDistortionMap[2];
RHISurface::ptr cameraMask[2];
cv::Mat cameraIntrinsicMatrix[2]; // From calibration
cv::Mat distCoeffs[2];
cv::Mat cameraOptimizedMatrix[2]; // Computed by cv::getOptimalNewCameraMatrix from cameraIntrinsicMatrix and distCoeffs

cv::Mat stereoRotation, stereoTranslation; // Calibrated
cv::Mat stereoRectification[2], stereoProjection[2]; // Derived from stereoRotation/stereoTranslation via cv::stereoRectify
cv::Mat stereoDisparityToDepth;
cv::Rect stereoValidROI[2];
static const cv::Mat zeroDistortion = cv::Mat::zeros(1, 5, CV_32FC1);


static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

void updateCameraDistortionMap(size_t cameraIdx, bool useStereoCalibration)  {
  cv::Size imageSize = cv::Size(s_cameraWidth, s_cameraHeight);
  float alpha = 0.25; // scaling factor. 0 = no invalid pixels in output (no black borders), 1 = use all input pixels
  cv::Mat map1, map2;
  cameraOptimizedMatrix[cameraIdx] = cv::getOptimalNewCameraMatrix(cameraIntrinsicMatrix[cameraIdx], distCoeffs[cameraIdx], imageSize, alpha, cv::Size(), NULL, /*centerPrincipalPoint=*/true);
  cv::initUndistortRectifyMap(cameraIntrinsicMatrix[cameraIdx], distCoeffs[cameraIdx], cv::noArray(), cameraOptimizedMatrix[cameraIdx], imageSize, CV_32F, map1, map2);

  if (useStereoCalibration) {
    cv::Mat stereo_map1, stereo_map2;

    // Compute the stereo remap and apply it on top of the intrinsic distortion remap layer
    cv::initUndistortRectifyMap(cameraOptimizedMatrix[cameraIdx], distCoeffs[cameraIdx], stereoRectification[cameraIdx], stereoProjection[cameraIdx], imageSize, CV_32F, stereo_map1, stereo_map2);

    cv::Mat remap1, remap2;
    cv::remap(map1, remap1, stereo_map1, stereo_map2, cv::INTER_LINEAR);
    cv::remap(map2, remap2, stereo_map1, stereo_map2, cv::INTER_LINEAR);

    map1 = remap1;
    map2 = remap2;
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

  for (size_t i = 0; i < 2; ++i) {
    printf("Projection matrix %zu:\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n  % .3f % .3f % .3f % .3f\n\n", i,
      eyeProjection[i][0][0], eyeProjection[i][0][1], eyeProjection[i][0][2], eyeProjection[i][0][3],
      eyeProjection[i][1][0], eyeProjection[i][1][1], eyeProjection[i][1][2], eyeProjection[i][1][3],
      eyeProjection[i][2][0], eyeProjection[i][2][1], eyeProjection[i][2][2], eyeProjection[i][2][3],
      eyeProjection[i][3][0], eyeProjection[i][3][1], eyeProjection[i][3][2], eyeProjection[i][3][3]);
  }


  // Cook the stereo separation transform into the projection matrices
  // TODO correct eye offsets
  eyeProjection[0] = eyeProjection[0] * glm::translate(glm::vec3(ipd *  stereoSeparationScale, 0.0f, 0.0f));
  eyeProjection[1] = eyeProjection[1] * glm::translate(glm::vec3(ipd * -stereoSeparationScale, 0.0f, 0.0f));

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

cv::Mat captureGreyscale(size_t cameraIdx, RHISurface::ptr tex, RHIRenderTarget::ptr rt, bool undistort) {

  rhi()->beginRenderPass(rt, kLoadInvalidate);
  if (undistort) {
    rhi()->bindRenderPipeline(camGreyscaleUndistortPipeline);
    rhi()->loadTexture(ksDistortionMap, cameraDistortionMap[cameraIdx], linearClampSampler);
  } else {
    rhi()->bindRenderPipeline(camGreyscalePipeline);
  }
  rhi()->loadTexture(ksImageTex, stereoCamera->rgbTexture(cameraIdx), linearClampSampler);
  rhi()->drawNDCQuad();
  rhi()->endRenderPass(rt);

  cv::Mat res;
  res.create(/*rows=*/ tex->height(), /*columns=*/tex->width(), CV_8UC1);
  assert(res.isContinuous());
  rhi()->readbackTexture(tex, 0, kVertexElementTypeUByte1N, res.ptr(0));
  return res;
}

void drawStatusLines(cv::Mat& image, const std::vector<std::string> lines) {
  std::vector<cv::Size> lineSizes; // total size of bounding rect per line
  std::vector<int> baselines; // Y size of area above baseline
  std::vector<int> lineYOffsets; // computed Y coordinate of line in drawing stack

  uint32_t rectPadding = 4;
  uint32_t linePaddingY = 4;
  double fontScale = 2.0;

  for (size_t lineIdx = 0; lineIdx < lines.size(); ++lineIdx) {
    int baseline = 0;
    lineSizes.push_back(cv::getTextSize(lines[lineIdx].c_str(), 1, fontScale, 1, &baseline));
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
    cv::Scalar(1, 1, 1), cv::FILLED);

  // Draw lines
  for (size_t lineIdx = 0; lineIdx < lines.size(); ++lineIdx) {
    cv::putText(image, lines[lineIdx].c_str(),
      cv::Point(
        origin.x + ((boundSize.width - lineSizes[lineIdx].width) / 2),
        origin.y + lineYOffsets[lineIdx] + lineSizes[lineIdx].height),
      1, fontScale, cv::Scalar(0, 255, 0));
  }
}

int main(int argc, char* argv[]) {

  startInputListenerThread();
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

  // Open the cameras
  stereoCamera = new ArgusCamera(demoState.display, demoState.context, {LEFT_CAMERA_INDEX, RIGHT_CAMERA_INDEX}, s_cameraWidth, s_cameraHeight, s_cameraFramerate);

  // Generate derived data for calibration
  s_charucoDictionary = cv::aruco::getPredefinedDictionary(s_charucoDictionaryName);
  s_charucoBoard = cv::aruco::CharucoBoard::create(s_charucoBoardSquareCountX, s_charucoBoardSquareCountY, s_charucoBoardSquareSideLengthMeters, s_charucoBoardMarkerSideLengthMeters, s_charucoDictionary);

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
          fs["camera0_matrix"] >> cameraIntrinsicMatrix[0];
          fs["camera1_matrix"] >> cameraIntrinsicMatrix[1];
          fs["camera0_distortionCoeffs"] >> distCoeffs[0];
          fs["camera1_distortionCoeffs"] >> distCoeffs[1];

          if (!(cameraIntrinsicMatrix[0].empty() || cameraIntrinsicMatrix[1].empty() || distCoeffs[0].empty() || distCoeffs[1].empty())) {
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
      // Textures and RTs we use for captures
      RHISurface::ptr fullGreyTex = rhi()->newTexture2D(s_cameraWidth, s_cameraHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));
      RHIRenderTarget::ptr fullGreyRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({fullGreyTex}));

      RHISurface::ptr feedbackTex = rhi()->newTexture2D(s_cameraWidth, s_cameraHeight, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));

      cv::Mat feedbackView;
      feedbackView.create(/*rows=*/ s_cameraHeight, /*columns=*/s_cameraWidth, CV_8UC4);

      for (unsigned int cameraIdx = 0; cameraIdx < 2; ++cameraIdx) {
retryIntrinsicCalibration:
        printf("Camera %u intrinsic calibration\n", cameraIdx);

        std::vector<cv::Mat> allCharucoCorners;
        std::vector<cv::Mat> allCharucoIds;

        while (!want_quit) {
          if (testButton(kButtonDown)) {
            // Calibration finished
            break;
          }

          bool found = false;

          stereoCamera->readFrame();
          cv::Mat viewFullRes = captureGreyscale(cameraIdx, fullGreyTex, fullGreyRT, /*undistort=*/false);


          std::vector<std::vector<cv::Point2f> > corners, rejected;
          std::vector<int> ids;
          cv::Mat currentCharucoCorners, currentCharucoIds;

          // Run ArUco marker detection
          cv::aruco::detectMarkers(viewFullRes, s_charucoDictionary, corners, ids, cv::aruco::DetectorParameters::create(), rejected);
          cv::aruco::refineDetectedMarkers(viewFullRes, s_charucoBoard, corners, ids, rejected);

          // Find corners using detected markers
          if (!ids.empty()) {
            cv::aruco::interpolateCornersCharuco(corners, ids, viewFullRes, s_charucoBoard, currentCharucoCorners, currentCharucoIds);
          }

          // Draw feedback points
          memset(feedbackView.ptr(0), 0, feedbackView.total() * 4);

          if (!ids.empty()) {
            cv::aruco::drawDetectedMarkers(feedbackView, corners);
          }
          if (currentCharucoCorners.total() > 3) {
            cv::aruco::drawDetectedCornersCharuco(feedbackView, currentCharucoCorners, currentCharucoIds);
          }

          // Require at least a third of the markers to be in frame to take an intrinsic calibration sample
          found = (currentCharucoCorners.total() >= (s_charucoBoard->chessboardCorners.size() / 3));

          char status1[64];
          char status2[64];
          sprintf(status1, "Camera %u (%s)", cameraIdx, cameraIdx == 0 ? "left" : "right");
          sprintf(status2, "%zu samples", allCharucoCorners.size());

          drawStatusLines(feedbackView, { status1, "Intrinsic calibration", status2 } );

          rhi()->loadTextureData(feedbackTex, kVertexElementTypeUByte4N, feedbackView.ptr(0));

          bool captureRequested = testButton(kButtonUp);
          if (found && captureRequested) {
  #ifdef SAVE_CALIBRATION_IMAGES
            char filename1[64];
            char filename2[64];
            static int fileIdx = 0;
            ++fileIdx;
            sprintf(filename1, "calib_%u_%02u_frame.png", cameraIdx, fileIdx);
            sprintf(filename2, "calib_%u_%02u_overlay.png", cameraIdx, fileIdx);

            //stbi_write_png(filename1, s_cameraWidth, s_cameraHeight, 1, viewFullRes.ptr(0), /*rowBytes=*/s_cameraWidth);
            //printf("Saved %s\n", filename1);

            // composite with the greyscale view and fix the alpha channel before writing
            for (size_t pixelIdx = 0; pixelIdx < (s_cameraWidth * s_cameraHeight); ++pixelIdx) {
              uint8_t* p = feedbackView.ptr(0) + (pixelIdx * 4);
              if (!(p[0] || p[1] || p[2])) {
                p[0] = p[1] = p[2] = viewFullRes.ptr(0)[pixelIdx];

              }
              p[3] = 0xff;
            }
            stbi_write_png(filename2, s_cameraWidth, s_cameraHeight, 4, feedbackView.ptr(0), /*rowBytes=*/s_cameraWidth * 4);
            printf("Saved %s\n", filename2);
  #endif

            allCharucoCorners.push_back(currentCharucoCorners);
            allCharucoIds.push_back(currentCharucoIds);
          }

          // Draw camera stream and feedback overlay to both eye RTs

          for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
            rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            rhi()->beginRenderPass(eyeRT[eyeIndex], kLoadClear);

            rhi()->bindRenderPipeline(camOverlayPipeline);
            rhi()->loadTexture(ksImageTex, stereoCamera->rgbTexture(cameraIdx), linearClampSampler);
            rhi()->loadTexture(ksOverlayTex, feedbackTex, linearClampSampler);

            // coordsys right now: -X = left, -Z = into screen
            // (camera is at the origin)
            const glm::vec3 tx = glm::vec3(0.0f, 0.0f, -7.0f);
            const float scaleFactor = 5.0f;
            glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(scaleFactor * (static_cast<float>(s_cameraWidth) / static_cast<float>(s_cameraHeight)), scaleFactor, 1.0f)); // TODO
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
        try {

          cv::Mat stdDeviations, perViewErrors;
          std::vector<float> reprojErrs;
          cv::Size imageSize(s_cameraWidth, s_cameraHeight);
          float aspectRatio = 1.0f;
          int flags = cv::CALIB_FIX_PRINCIPAL_POINT | cv::CALIB_FIX_ASPECT_RATIO;

          cameraIntrinsicMatrix[cameraIdx] = cv::Mat::eye(3, 3, CV_64F);
          if( flags & cv::CALIB_FIX_ASPECT_RATIO )
              cameraIntrinsicMatrix[cameraIdx].at<double>(0,0) = aspectRatio;
          distCoeffs[cameraIdx] = cv::Mat::zeros(8, 1, CV_64F);


          double rms = cv::aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds,
                                         s_charucoBoard, imageSize,
                                         cameraIntrinsicMatrix[cameraIdx], distCoeffs[cameraIdx],
                                         cv::noArray(), cv::noArray(), stdDeviations, cv::noArray(),
                                         perViewErrors, flags);


          printf("RMS error reported by calibrateCameraCharuco: %g\n", rms);
          std::cout << "Camera " << cameraIdx << " Per-view error: " << std::endl << perViewErrors << std::endl;
          std::cout << "Camera " << cameraIdx << " Matrix: " << std::endl << cameraIntrinsicMatrix[cameraIdx] << std::endl;
          std::cout << "Camera " << cameraIdx << " Distortion coefficients: " << std::endl << distCoeffs[cameraIdx] << std::endl;

          // Build initial intrinsic-only distortion map
          updateCameraDistortionMap(cameraIdx, false);
        } catch (const std::exception& ex) {
          printf("Camera intrinsic calibration failed: %s\n", ex.what());
          goto retryIntrinsicCalibration;
        }

        // Show a preview of the intrinsic calibration and give the option to retry or continue
        {
          RHISurface::ptr tempMask = rhi()->newTexture2D(4, 4, RHISurfaceDescriptor(kSurfaceFormat_R8));
          uint8_t tempMaskData[16];
          memset(tempMaskData, 0xff, 16);
          rhi()->loadTextureData(tempMask, kVertexElementTypeUByte1N, tempMaskData);

          while (!want_quit) {
            if (testButton(kButtonLeft)) {
              goto retryIntrinsicCalibration;
            }
            if (testButton(kButtonRight)) {
              // Calibration accepted by user
              break;
            }

            stereoCamera->readFrame();

            for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
              rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
              rhi()->beginRenderPass(eyeRT[eyeIndex], kLoadClear);

              rhi()->bindRenderPipeline(camUndistortMaskPipeline);
              rhi()->loadTexture(ksImageTex, stereoCamera->rgbTexture(cameraIdx), linearClampSampler);
              rhi()->loadTexture(ksDistortionMap, cameraDistortionMap[cameraIdx], linearClampSampler);
              rhi()->loadTexture(ksMaskTex, tempMask, linearClampSampler);

              // coordsys right now: -X = left, -Z = into screen
              // (camera is at the origin)
              const glm::vec3 tx = glm::vec3(0.0f, 0.0f, -7.0f);
              const float scaleFactor = 5.0f;
              glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(scaleFactor * (static_cast<float>(s_cameraWidth) / static_cast<float>(s_cameraHeight)), scaleFactor, 1.0f)); // TODO
              glm::mat4 mvp = eyeProjection[eyeIndex] * model;

              NDCClippedQuadUniformBlock ub;
              ub.modelViewProjection = mvp;
              ub.minUV = glm::vec2(0.0f);
              ub.maxUV = glm::vec2(1.0f);

              rhi()->loadUniformBlockImmediate(ksNDCClippedQuadUniformBlock, &ub, sizeof(NDCClippedQuadUniformBlock));

              rhi()->drawNDCQuad();

              rhi()->endRenderPass(eyeRT[eyeIndex]);
            }
            renderHMDFrame();
          } // Preview rendering
        }

      } // Per-camera calibration loop

    } // Individual camera calibration

    // Incrementally save calibration data if it was updated this run
    // TODO: should probably remove this after finished debugging stereo calibration
    if (want_quit)
      goto quit;

    if (needIntrinsicCalibration) {
      cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
      fs.write("camera0_matrix", cameraIntrinsicMatrix[0]);
      fs.write("camera1_matrix", cameraIntrinsicMatrix[1]);
      fs.write("camera0_distortionCoeffs", distCoeffs[0]);
      fs.write("camera1_distortionCoeffs", distCoeffs[1]);
      printf("Saved updated intrinsic calibration data\n");
    }

    if (needStereoCalibration) {
retryStereoCalibration:
      // Stereo pair calibration

      // Textures and RTs we use for full-res captures.
      RHISurface::ptr fullGreyTex[2];
      RHIRenderTarget::ptr fullGreyRT[2];
      RHISurface::ptr feedbackTex[2];

      for (size_t viewIdx = 0; viewIdx < 2; ++viewIdx) {
        fullGreyTex[viewIdx] = rhi()->newTexture2D(s_cameraWidth, s_cameraHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));
        fullGreyRT[viewIdx] = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({fullGreyTex[viewIdx]}));

        feedbackTex[viewIdx] = rhi()->newTexture2D(s_cameraWidth, s_cameraHeight, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
      }

      cv::Mat feedbackView[2];
      feedbackView[0].create(/*rows=*/ s_cameraHeight, /*columns=*/s_cameraWidth, CV_8UC4);
      feedbackView[1].create(/*rows=*/ s_cameraHeight, /*columns=*/s_cameraWidth, CV_8UC4);

      std::vector<std::vector<cv::Point3f> > objectPoints; // Points from the board definition for the relevant corners each frame
      std::vector<std::vector<cv::Point2f> > calibrationPoints[2]; // Points in image space for the 2 views for the relevant corners each frame

      while (!want_quit) {
        if (testButton(kButtonDown)) {
          // Calibration finished
          break;
        }

        stereoCamera->readFrame();

        // Capture and undistort camera views.
        cv::Mat viewFullRes[2];
        viewFullRes[0] = captureGreyscale(0, fullGreyTex[0], fullGreyRT[0], /*undistort=*/true);
        viewFullRes[1] = captureGreyscale(1, fullGreyTex[1], fullGreyRT[1], /*undistort=*/true);


        std::vector<std::vector<cv::Point2f> > corners[2], rejected[2];
        std::vector<int> ids[2];

        std::vector<cv::Point2f> currentCharucoCornerPoints[2];
        std::vector<int> currentCharucoCornerIds[2];

        // Run ArUco marker detection
        // Note that we don't feed the camera distortion parameters to the aruco functions here, since the views we're operating on have already been undistorted.
        for (size_t viewIdx = 0; viewIdx < 2; ++viewIdx) {
          cv::aruco::detectMarkers(viewFullRes[viewIdx], s_charucoDictionary, corners[viewIdx], ids[viewIdx], cv::aruco::DetectorParameters::create(), rejected[viewIdx], cameraOptimizedMatrix[viewIdx], zeroDistortion);
          cv::aruco::refineDetectedMarkers(viewFullRes[viewIdx], s_charucoBoard, corners[viewIdx], ids[viewIdx], rejected[viewIdx], cameraOptimizedMatrix[viewIdx], zeroDistortion);

          // Find chessboard corners using detected markers
          if (!ids[viewIdx].empty()) {
            cv::aruco::interpolateCornersCharuco(corners[viewIdx], ids[viewIdx], viewFullRes[viewIdx], s_charucoBoard, currentCharucoCornerPoints[viewIdx], currentCharucoCornerIds[viewIdx], cameraOptimizedMatrix[viewIdx], zeroDistortion);
          }
        }

        // Find set of chessboard corners present in both views
        std::set<int> stereoCornerIds;
        {
          std::set<int> view0Ids;
          for (size_t i = 0; i < currentCharucoCornerIds[0].size(); ++i) {
            view0Ids.insert(currentCharucoCornerIds[0][i]);
          }
          for (size_t i = 0; i < currentCharucoCornerIds[1].size(); ++i) {
            int id = currentCharucoCornerIds[1][i];
            if (view0Ids.find(id) != view0Ids.end())
              stereoCornerIds.insert(id);
          }
        }

        // Require at least 6 corners visibile to both cameras to consider this frame
        bool foundOverlap = stereoCornerIds.size() >= 6;

        // Filter the view corner sets to only overlapping corners, which we will later feed to stereoCalibrate
        std::vector<cv::Point3f> thisFrameBoardRefCorners;
        std::vector<cv::Point2f> thisFrameImageCorners[2];

        for (std::set<int>::const_iterator corner_it = stereoCornerIds.begin(); corner_it != stereoCornerIds.end(); ++corner_it) {
          int cornerId = *corner_it;

          for (size_t viewIdx = 0; viewIdx < 2; ++viewIdx) {
            for (int viewCornerIdx = 0; viewCornerIdx < currentCharucoCornerIds[viewIdx].size(); ++viewCornerIdx) {
              if (currentCharucoCornerIds[viewIdx][viewCornerIdx] == cornerId) {
                thisFrameImageCorners[viewIdx].push_back(currentCharucoCornerPoints[viewIdx][viewCornerIdx]);
                break;
              }
            }
          }

          // save the corner point in board space from the board definition
          thisFrameBoardRefCorners.push_back(s_charucoBoard->chessboardCorners[cornerId]);
        }
        assert(thisFrameBoardRefCorners.size() == thisFrameImageCorners[0].size() && thisFrameBoardRefCorners.size() == thisFrameImageCorners[1].size());

        bool captureRequested = testButton(kButtonUp);
        if (foundOverlap && captureRequested) {

#if 0 //def SAVE_CALIBRATION_IMAGES
          char filename1[64];
          char filename2[64];
          static int fileIdx = 0;
          ++fileIdx;
          sprintf(filename1, "calib_stereo_%02u_frame.png", fileIdx);
          sprintf(filename2, "calib_stereo_%02u_overlay.png", fileIdx);

          //stbi_write_png(filename1, s_cameraWidth*2, s_cameraHeight, 1, viewFullResStereo.ptr(0), /*rowBytes=*/(s_cameraWidth*2));
          //printf("Saved %s\n", filename2);

          // composite with the greyscale view and fix the alpha channel before writing
          for (size_t pixelIdx = 0; pixelIdx < (s_cameraWidth * 2) * s_cameraHeight; ++pixelIdx) {
            uint8_t* p = feedbackViewStereo.ptr(0) + (pixelIdx * 4);
            if (!(p[0] || p[1] || p[2])) {
              p[0] = p[1] = p[2] = viewFullResStereo.ptr(0)[pixelIdx];

            }
            p[3] = 0xff;
          }
          stbi_write_png(filename2, s_cameraWidth*2, s_cameraHeight, 4, feedbackViewStereo.ptr(0), /*rowBytes=*/s_cameraWidth*2 * 4);
          printf("Saved %s\n", filename2);
#endif

          objectPoints.push_back(thisFrameBoardRefCorners);
          calibrationPoints[0].push_back(thisFrameImageCorners[0]);
          calibrationPoints[1].push_back(thisFrameImageCorners[1]);
        }

        // Draw feedback points
        for (size_t viewIdx = 0; viewIdx < 2; ++viewIdx) {
          memset(feedbackView[viewIdx].ptr(0), 0, feedbackView[viewIdx].total() * 4);

          if (!corners[viewIdx].empty()) {
            cv::aruco::drawDetectedMarkers(feedbackView[viewIdx], corners[viewIdx]);
          }

          // Borrowed from cv::aruco::drawDetectedCornersCharuco -- modified to switch the color per-marker to indicate stereo visibility
          for(size_t cornerIdx = 0; cornerIdx < currentCharucoCornerIds[viewIdx].size(); ++cornerIdx) {
            cv::Point2f corner = currentCharucoCornerPoints[viewIdx][cornerIdx];
            int id = currentCharucoCornerIds[viewIdx][cornerIdx];

            // grey for mono points
            cv::Scalar cornerColor = cv::Scalar(127, 127, 127);
            if (stereoCornerIds.find(id) != stereoCornerIds.end()) {
              // red for stereo points
              cornerColor = cv::Scalar(255, 0, 0);
            }

            // draw first corner mark
            cv::rectangle(feedbackView[viewIdx], corner - cv::Point2f(3, 3), corner + cv::Point2f(3, 3), cornerColor, 1, cv::LINE_AA);

            // draw ID
            char idbuf[16];
            sprintf(idbuf, "id=%u", id);
            cv::putText(feedbackView[viewIdx], idbuf, corner + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cornerColor, 2);
          }
        }

        char status1[64];
        sprintf(status1, "%zu samples", calibrationPoints[0].size());

        for (size_t viewIdx = 0; viewIdx < 2; ++viewIdx)
          drawStatusLines(feedbackView[viewIdx], { "Stereo calibration", status1 } );

        rhi()->loadTextureData(feedbackTex[0], kVertexElementTypeUByte4N, feedbackView[0].ptr(0));
        rhi()->loadTextureData(feedbackTex[1], kVertexElementTypeUByte4N, feedbackView[1].ptr(0));


        // Draw camera stream and feedback overlay to both eye RTs

        for (int eyeIndex = 0; eyeIndex < 2; ++eyeIndex) {
          rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          rhi()->beginRenderPass(eyeRT[eyeIndex], kLoadClear);

          rhi()->bindRenderPipeline(camOverlayStereoUndistortPipeline);
          rhi()->loadTexture(ksLeftCameraTex, stereoCamera->rgbTexture(0), linearClampSampler);
          rhi()->loadTexture(ksRightCameraTex, stereoCamera->rgbTexture(1), linearClampSampler);
          rhi()->loadTexture(ksLeftOverlayTex, feedbackTex[0], linearClampSampler);
          rhi()->loadTexture(ksRightOverlayTex, feedbackTex[1], linearClampSampler);
          rhi()->loadTexture(ksLeftDistortionMap, cameraDistortionMap[0], linearClampSampler);
          rhi()->loadTexture(ksRightDistortionMap, cameraDistortionMap[1], linearClampSampler);

          // coordsys right now: -X = left, -Z = into screen
          // (camera is at the origin)
          const glm::vec3 tx = glm::vec3(0.0f, 0.0f, -7.0f);
          const float scaleFactor = 2.5f;
          glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(scaleFactor * (static_cast<float>(s_cameraWidth*2) / static_cast<float>(s_cameraHeight)), scaleFactor, 1.0f)); // TODO
          glm::mat4 mvp = eyeProjection[eyeIndex] * model;

          NDCQuadUniformBlock ub;
          ub.modelViewProjection = mvp;
          rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(NDCQuadUniformBlock));

          rhi()->drawNDCQuad();

          rhi()->endRenderPass(eyeRT[eyeIndex]);
        }
        renderHMDFrame();

      } // Stereo calibration sample-gathering loop

      if (want_quit)
        goto quit;

      // Samples collected, run calibration
      cv::Mat E, F;
      cv::Mat perViewErrors;

      try {
        // Note the use of the zero distortion matrix and optimized camera matrix, since we already corrected for individual camera distortions when capturing the images.

        double rms = cv::stereoCalibrate(objectPoints,
          calibrationPoints[0], calibrationPoints[1],
          cameraOptimizedMatrix[0], zeroDistortion,
          cameraOptimizedMatrix[1], zeroDistortion,
          cv::Size(s_cameraWidth, s_cameraHeight),
          stereoRotation, stereoTranslation, E, F, perViewErrors, cv::CALIB_FIX_INTRINSIC);

        printf("RMS error reported by stereoCalibrate: %g\n", rms);
        std::cout << " Per-view error: " << std::endl << perViewErrors << std::endl;
      } catch (const std::exception& ex) {
        printf("Stereo calibration failed: %s\n", ex.what());
        goto retryStereoCalibration;
      }

    } // needStereoCalibration

    if (want_quit)
      goto quit;

    // Compute rectification/projection transforms from the stereo calibration data
    float alpha = -1.0f;  //0.25;

    // Using the optimized camera matrix and zero distortion matrix again to create a rectification remap that can be layered on top of the intrinsic distortion remap
    cv::stereoRectify(
      cameraOptimizedMatrix[0], zeroDistortion,
      cameraOptimizedMatrix[1], zeroDistortion,
      cv::Size(s_cameraWidth, s_cameraHeight),
      stereoRotation, stereoTranslation,
      stereoRectification[0], stereoRectification[1],
      stereoProjection[0], stereoProjection[1],
      stereoDisparityToDepth,
      /*flags=*/cv::CALIB_ZERO_DISPARITY, alpha, cv::Size(),
      &stereoValidROI[0], &stereoValidROI[1]);

    // Camera info dump
    for (size_t cameraIdx = 0; cameraIdx < 2; ++cameraIdx) {
      printf("\n ===== Camera %zu ===== \n", cameraIdx);

      std::cout << "* Rectification matrix:" << std::endl << stereoRectification[cameraIdx] << std::endl;
      std::cout << "* Projection matrix:" << std::endl << stereoProjection[cameraIdx] << std::endl;
      std::cout << "* Valid image region:" << std::endl << stereoValidROI[cameraIdx] << std::endl;


      const double apertureSize = 6.35; // mm, 1/4" sensor
      double fovX, fovY, focalLength, aspectRatio;
      cv::Point2d principalPoint;

      cv::calibrationMatrixValues(cameraIntrinsicMatrix[cameraIdx], cv::Size(s_cameraWidth, s_cameraHeight), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
      printf("* Intrinsic matrix: FOV %.1f x %.1f deg, approx focal length %.2fmm\n", fovX, fovY, focalLength);
      cv::calibrationMatrixValues(cameraOptimizedMatrix[cameraIdx], cv::Size(s_cameraWidth, s_cameraHeight), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
      printf("* Optimized matrix: FOV %.1f x %.1f deg\n", fovX, fovY);
      cv::calibrationMatrixValues(cv::Mat(stereoProjection[cameraIdx], cv::Rect(0, 0, 3, 3)), cv::Size(s_cameraWidth, s_cameraHeight), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
      printf("* Stereo projection matrix: FOV %.1f x %.1f deg\n", fovX, fovY);
    }
    printf("\n ==================== \n");

    // Check the valid image regions for a failed stereo calibration. A bad calibration will usually result in a valid ROI for one or both views with a 0-pixel dimension.
/*
    if (needStereoCalibration) {
      if (stereoValidROI[0].area() == 0 || stereoValidROI[1].area() == 0) {
        printf("Stereo calibration failed: one or both of the valid image regions has zero area.\n");
        goto retryStereoCalibration;
      }
    }
*/

    // Save calibration data if it was updated this run
    if (needStereoCalibration || needIntrinsicCalibration) {
      cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
      fs.write("camera0_matrix", cameraIntrinsicMatrix[0]);
      fs.write("camera1_matrix", cameraIntrinsicMatrix[1]);
      fs.write("camera0_distortionCoeffs", distCoeffs[0]);
      fs.write("camera1_distortionCoeffs", distCoeffs[1]);
      fs.write("stereoRotation", stereoRotation);
      fs.write("stereoTranslation", stereoTranslation);
      printf("Saved updated calibration data\n");
    }


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
        STBI_FREE(maskData);
        maskData = NULL;
      }

      if (!maskData) {
        printf("No usable mask data found in \"%s\" for camera %zu. A template will be created.\n", filename, cameraIdx);

        x = s_cameraWidth;
        y = s_cameraHeight;
        maskData = (uint8_t*) STBI_MALLOC(x * y);

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

      STBI_FREE(maskData);
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
        boost::accumulators::tag::rolling_mean
      > > frameInterval(boost::accumulators::tag::rolling_window::window_size = 1024);

    while (!want_quit) {
      ++frameCounter;
      {
        // Scale factor adjustment

        bool didChangeScale = false;
        if (testButton(kButtonUp)) {
          scaleFactor += 0.1f;
          didChangeScale = true;
        }
        if (testButton(kButtonDown)) {
          scaleFactor -= 0.1f;
          didChangeScale = true;
        }

        if (didChangeScale) {
          printf("New scale factor: %.3g\n", scaleFactor);
        }

#if 0
        bool didChangeStereoSeparation = false;
        if (testButton(kButtonRight)) {
          stereoSeparationScale += 0.1f;
          didChangeStereoSeparation = true;
        }
        if (testButton(kButtonLeft)) {
          stereoSeparationScale -= 0.1f;
          didChangeStereoSeparation = true;
        }

        if (didChangeStereoSeparation) {
          printf("New stereo separation scale: %.3g\n", stereoSeparationScale);
          recomputeHMDParameters();
        }
#endif

      }

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

        printf("Frame interval: % .6f ms (% .6f fps)\n",
          static_cast<double>(boost::accumulators::rolling_mean(frameInterval)) / 1000000.0,
          1000000000.0 / static_cast<double>(boost::accumulators::rolling_mean(frameInterval)));
        //printf("Frame interval: %f ns\n", boost::accumulators::rolling_mean(frameInterval));

        captureInterval = {};

        //printf("CLOCK_MONOTONIC: %llu. Sensor timestamps: %llu %llu\n", raw_ns, stereoCamera->sensorTimestamp(0), stereoCamera->sensorTimestamp(1));
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
        const glm::vec3 tx = glm::vec3(0.0f, 0.0f, -7.0f);
        glm::mat4 model = glm::translate(tx) * glm::scale(glm::vec3(scaleFactor * (static_cast<float>(stereoCamera->streamWidth()) / static_cast<float>(stereoCamera->streamHeight())), scaleFactor, 1.0f)); // TODO
        glm::mat4 mvp = eyeProjection[eyeIndex] * model;

        NDCClippedQuadUniformBlock ub;
        ub.modelViewProjection = mvp;
        if (eyeIndex == 0) { // left
          ub.minUV = glm::vec2(0.0f,  0.0f);
          ub.maxUV = glm::vec2(0.75f, 1.0f);
        } else { // right
          ub.minUV = glm::vec2(0.25f, 0.0f);
          ub.maxUV = glm::vec2(1.0f,  1.0f);
        }

        rhi()->loadUniformBlockImmediate(ksNDCClippedQuadUniformBlock, &ub, sizeof(NDCClippedQuadUniformBlock));

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

