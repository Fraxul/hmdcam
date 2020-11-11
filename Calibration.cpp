#include "Calibration.h"
#include "Render.h"
#include <set>
#include <vector>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "ArgusCamera.h"
#include "InputListener.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>

// #define SAVE_CALIBRATION_IMAGES
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

// Imported data
extern size_t s_cameraWidth, s_cameraHeight;
extern bool want_quit;
extern RHISurface::ptr eyeTex[2];
extern RHIRenderTarget::ptr eyeRT[2];
extern glm::mat4 eyeProjection[2];

extern float scaleFactor;
extern float stereoSeparationScale;
extern FxAtomicString ksImageTex;
extern FxAtomicString ksLeftCameraTex;
extern FxAtomicString ksRightCameraTex;
extern FxAtomicString ksLeftDistortionMap;
extern FxAtomicString ksRightDistortionMap;
extern FxAtomicString ksOverlayTex;
extern FxAtomicString ksLeftOverlayTex;
extern FxAtomicString ksRightOverlayTex;
extern FxAtomicString ksDistortionMap;
extern FxAtomicString ksMaskTex;
extern RHIRenderPipeline::ptr camOverlayPipeline;
extern RHIRenderPipeline::ptr camOverlayStereoPipeline;
extern RHIRenderPipeline::ptr camOverlayStereoUndistortPipeline;
extern RHIRenderPipeline::ptr camUndistortMaskPipeline;
extern RHIRenderPipeline::ptr camGreyscalePipeline;
extern RHIRenderPipeline::ptr camGreyscaleUndistortPipeline;

// Camera info/state
extern ArgusCamera* stereoCamera;
extern RHISurface::ptr cameraDistortionMap[2];
extern RHISurface::ptr cameraMask[2];
cv::Mat cameraIntrinsicMatrix[2]; // From calibration
cv::Mat distCoeffs[2];
cv::Mat cameraOptimizedMatrix[2]; // Computed by cv::getOptimalNewCameraMatrix from cameraIntrinsicMatrix and distCoeffs

cv::Mat stereoRotation, stereoTranslation; // Calibrated
cv::Mat stereoRectification[2], stereoProjection[2]; // Derived from stereoRotation/stereoTranslation via cv::stereoRectify
cv::Mat stereoDisparityToDepth;
cv::Rect stereoValidROI[2];
static const cv::Mat zeroDistortion = cv::Mat::zeros(1, 5, CV_32FC1);

static bool s_haveIntrinsicCalibration = false;
static bool s_haveStereoCalibration = false;

// ChArUco target pattern config
const cv::aruco::PREDEFINED_DICTIONARY_NAME s_charucoDictionaryName = cv::aruco::DICT_5X5_100;
const unsigned int s_charucoBoardSquareCountX = 12;
const unsigned int s_charucoBoardSquareCountY = 9;
const float s_charucoBoardSquareSideLengthMeters = 0.060f;
const float s_charucoBoardMarkerSideLengthMeters = 0.045f;

static cv::Ptr<cv::aruco::Dictionary> s_charucoDictionary;
static cv::Ptr<cv::aruco::CharucoBoard> s_charucoBoard;

// ----


cv::Mat captureGreyscale(size_t cameraIdx, RHISurface::ptr tex, RHIRenderTarget::ptr rt, bool undistort);
void drawStatusLines(cv::Mat& image, const std::vector<std::string> lines);

// ----

void initCalibration() {
  s_charucoDictionary = cv::aruco::getPredefinedDictionary(s_charucoDictionaryName);
  s_charucoBoard = cv::aruco::CharucoBoard::create(s_charucoBoardSquareCountX, s_charucoBoardSquareCountY, s_charucoBoardSquareSideLengthMeters, s_charucoBoardMarkerSideLengthMeters, s_charucoDictionary);
}


void readCalibrationData() {
  s_haveIntrinsicCalibration = false;
  s_haveStereoCalibration = false;

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
        s_haveIntrinsicCalibration = true;
      }

      fs["stereoRotation"] >> stereoRotation;
      fs["stereoTranslation"] >> stereoTranslation;
      if (!(stereoRotation.empty() || stereoTranslation.empty())) {
        s_haveStereoCalibration = true;
        printf("Loaded stereo offset calibration data from file\n");
      }

    } catch (const std::exception& ex) {
      printf("Unable to read calibration data: %s\n", ex.what());
    }

  } else {
    printf("Unable to open calibration data file\n");
  }
}

void saveCalibrationData() {
  assert(s_haveIntrinsicCalibration);

  cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
  if (s_haveIntrinsicCalibration) {
    fs.write("camera0_matrix", cameraIntrinsicMatrix[0]);
    fs.write("camera1_matrix", cameraIntrinsicMatrix[1]);
    fs.write("camera0_distortionCoeffs", distCoeffs[0]);
    fs.write("camera1_distortionCoeffs", distCoeffs[1]);
  }
  if (s_haveStereoCalibration) {
    fs.write("stereoRotation", stereoRotation);
    fs.write("stereoTranslation", stereoTranslation);
  }
  printf("Saved updated calibration data\n");
}

bool haveIntrinsicCalibration() { return s_haveIntrinsicCalibration; }
bool haveStereoCalibration() { return s_haveStereoCalibration; }

void updateCameraDistortionMap(size_t cameraIdx, bool useStereoCalibration)  {
  cv::Size imageSize = cv::Size(s_cameraWidth, s_cameraHeight);
  float alpha = 0.25; // scaling factor. 0 = no invalid pixels in output (no black borders), 1 = use all input pixels
  cv::Mat map1, map2;
  if (useStereoCalibration) {
    // Stereo remap -- apply both distortion correction and stereo rectification/projection
    cv::initUndistortRectifyMap(cameraIntrinsicMatrix[cameraIdx], distCoeffs[cameraIdx], stereoRectification[cameraIdx], stereoProjection[cameraIdx], imageSize, CV_32F, map1, map2);
  } else {
    // Intrinic-only remap -- no rectification transform, using optimized matrix
    cameraOptimizedMatrix[cameraIdx] = cv::getOptimalNewCameraMatrix(cameraIntrinsicMatrix[cameraIdx], distCoeffs[cameraIdx], imageSize, alpha, cv::Size(), NULL, /*centerPrincipalPoint=*/true);
    cv::initUndistortRectifyMap(cameraIntrinsicMatrix[cameraIdx], distCoeffs[cameraIdx], cv::noArray(), cameraOptimizedMatrix[cameraIdx], imageSize, CV_32F, map1, map2);
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

void doIntrinsicCalibration() {
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

    clearButtons();
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

      // Debug stream rendering
      {
        RHISurface::ptr debugSurface = renderAcquireDebugSurface();
        if (debugSurface) {
          rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
          RHIRenderTarget::ptr rt = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({debugSurface}));
          rhi()->beginRenderPass(rt, kLoadClear);

          // Center the feedback on the debug surface, which should be twice as wide as the camera view.
          RHIRect vp = RHIRect::xywh(debugSurface->width() / 4, 0, debugSurface->width() / 2, debugSurface->height());
          rhi()->setViewport(vp);

          rhi()->bindRenderPipeline(camOverlayPipeline);
          rhi()->loadTexture(ksImageTex, stereoCamera->rgbTexture(cameraIdx), linearClampSampler);
          rhi()->loadTexture(ksOverlayTex, feedbackTex, linearClampSampler);

          NDCQuadUniformBlock ub;
          ub.modelViewProjection = glm::mat4(1.0f);
          rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(NDCQuadUniformBlock));
          rhi()->drawNDCQuad();

          rhi()->endRenderPass(rt);
          renderSubmitDebugSurface(debugSurface);
        }
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
      clearButtons();
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
          rhi()->loadTexture(ksMaskTex, disabledMaskTex, linearClampSampler);

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

        // Debug stream rendering
        {
          RHISurface::ptr debugSurface = renderAcquireDebugSurface();
          if (debugSurface) {
            rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
            RHIRenderTarget::ptr rt = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({debugSurface}));
            rhi()->beginRenderPass(rt, kLoadClear);

            // Center the feedback on the debug surface, which should be twice as wide as the camera view.
            RHIRect vp = RHIRect::xywh(debugSurface->width() / 4, 0, debugSurface->width() / 2, debugSurface->height());
            rhi()->setViewport(vp);

            rhi()->bindRenderPipeline(camUndistortMaskPipeline);
            rhi()->loadTexture(ksImageTex, stereoCamera->rgbTexture(cameraIdx), linearClampSampler);
            rhi()->loadTexture(ksDistortionMap, cameraDistortionMap[cameraIdx], linearClampSampler);
            rhi()->loadTexture(ksMaskTex, disabledMaskTex, linearClampSampler);

            NDCClippedQuadUniformBlock ub;
            ub.modelViewProjection = glm::mat4(1.0f);
            ub.minUV = glm::vec2(0.0f);
            ub.maxUV = glm::vec2(1.0f);

            rhi()->loadUniformBlockImmediate(ksNDCClippedQuadUniformBlock, &ub, sizeof(NDCClippedQuadUniformBlock));

            rhi()->drawNDCQuad();

            rhi()->endRenderPass(rt);
            renderSubmitDebugSurface(debugSurface);
          }
        }

        renderHMDFrame();
      } // Preview rendering
    }

  } // Per-camera calibration loop

  s_haveIntrinsicCalibration = true;
}


void doStereoCalibration() {
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

  clearButtons();
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
        for (size_t viewCornerIdx = 0; viewCornerIdx < currentCharucoCornerIds[viewIdx].size(); ++viewCornerIdx) {
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

    // Debug stream rendering
    {
      RHISurface::ptr debugSurface = renderAcquireDebugSurface();
      if (debugSurface) {
        rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        RHIRenderTarget::ptr rt = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({debugSurface}));
        rhi()->beginRenderPass(rt, kLoadClear);

        rhi()->bindRenderPipeline(camOverlayStereoUndistortPipeline);
        rhi()->loadTexture(ksLeftCameraTex, stereoCamera->rgbTexture(0), linearClampSampler);
        rhi()->loadTexture(ksRightCameraTex, stereoCamera->rgbTexture(1), linearClampSampler);
        rhi()->loadTexture(ksLeftOverlayTex, feedbackTex[0], linearClampSampler);
        rhi()->loadTexture(ksRightOverlayTex, feedbackTex[1], linearClampSampler);
        rhi()->loadTexture(ksLeftDistortionMap, cameraDistortionMap[0], linearClampSampler);
        rhi()->loadTexture(ksRightDistortionMap, cameraDistortionMap[1], linearClampSampler);

        NDCQuadUniformBlock ub;
        // Don't need a projection, just fill the surface with the feedback quad.
        ub.modelViewProjection = glm::mat4(1.0f);
        rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(NDCQuadUniformBlock));

        rhi()->drawNDCQuad();

        rhi()->endRenderPass(rt);
        renderSubmitDebugSurface(debugSurface);
      }
    }

    renderHMDFrame();

  } // Stereo calibration sample-gathering loop

  if (want_quit)
    return;

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

  s_haveStereoCalibration = true;
}



void generateCalibrationDerivedData() {
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
  std::cout << "Stereo translation:" << std::endl << stereoTranslation << std::endl;
  std::cout << "Stereo rotation matrix:" << std::endl << stereoRotation << std::endl;


  // Check the valid image regions for a failed stereo calibration. A bad calibration will usually result in a valid ROI for one or both views with a 0-pixel dimension.
/*
  if (needStereoCalibration) {
    if (stereoValidROI[0].area() == 0 || stereoValidROI[1].area() == 0) {
      printf("Stereo calibration failed: one or both of the valid image regions has zero area.\n");
      goto retryStereoCalibration;
    }
  }
*/
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

