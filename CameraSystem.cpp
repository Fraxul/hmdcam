#include "CameraSystem.h"
#include "ArgusCamera.h"
#include "rhi/RHIResources.h"
#include "Render.h"
#include "imgui.h"
#include <iostream>
#include <set>
#include <assert.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

static const cv::Mat zeroDistortion = cv::Mat::zeros(1, 5, CV_32FC1);

// ChAruCo target pattern config
const cv::aruco::PREDEFINED_DICTIONARY_NAME s_charucoDictionaryName = cv::aruco::DICT_5X5_100;
const unsigned int s_charucoBoardSquareCountX = 12;
const unsigned int s_charucoBoardSquareCountY = 9;
const float s_charucoBoardSquareSideLengthMeters = 0.060f;
const float s_charucoBoardMarkerSideLengthMeters = 0.045f;

static cv::Ptr<cv::aruco::Dictionary> s_charucoDictionary;
static cv::Ptr<cv::aruco::CharucoBoard> s_charucoBoard;







CameraSystem::CameraSystem(ArgusCamera* ac) : calibrationFilename("calibration.yml"), m_argusCamera(ac) {

  // Initialize ChAruCo data on first use
  if (!s_charucoDictionary)
    s_charucoDictionary = cv::aruco::getPredefinedDictionary(s_charucoDictionaryName);

  if (!s_charucoBoard)
    s_charucoBoard = cv::aruco::CharucoBoard::create(s_charucoBoardSquareCountX, s_charucoBoardSquareCountY, s_charucoBoardSquareSideLengthMeters, s_charucoBoardMarkerSideLengthMeters, s_charucoDictionary);

  m_cameras.resize(m_argusCamera->streamCount());

}

bool CameraSystem::loadCalibrationData() {

  cv::FileStorage fs(calibrationFilename.c_str(), cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    printf("Unable to open calibration data file\n");
    return false;
  }

  try {
    cv::FileNode camerasFn = fs["cameras"];
    if (camerasFn.isSeq()) {
      for (size_t cameraIdx = 0; cameraIdx < m_cameras.size(); ++cameraIdx) {
        cv::FileNode cfn = camerasFn[cameraIdx];
        if (cfn.isMap()) {
          Camera& c = m_cameras[cameraIdx];

          cfn["intrinsicMatrix"] >> c.intrinsicMatrix;
          cfn["distortionCoeffs"] >> c.distCoeffs;
        }
      }
    }

    cv::FileNode viewsFn = fs["views"];
    if (viewsFn.isSeq()) {
      m_views.resize(viewsFn.size());

      for (size_t viewIdx = 0; viewIdx < viewsFn.size(); ++viewIdx) {
        cv::FileNode vfn = viewsFn[viewIdx];
        if (vfn.isMap()) {
          View& v = m_views[viewIdx];

          vfn["isStereo"] >> v.isStereo;

          if (v.isStereo) {
            vfn["leftCameraIndex"] >> v.cameraIndices[0];
            vfn["rightCameraIndex"] >> v.cameraIndices[1];
            vfn["stereoRotation"] >> v.stereoRotation;
            vfn["stereoTranslation"] >> v.stereoTranslation;
          } else {
            vfn["cameraIndex"] >> v.cameraIndices[0];
            v.cameraIndices[1] = v.cameraIndices[0];
          }
        }
      }
    }

  } catch (const std::exception& ex) {
    printf("Unable to load calibration data: %s\n", ex.what());
    return false;
  }

  for (size_t cameraIdx = 0; cameraIdx < m_cameras.size(); ++cameraIdx) {
    Camera& c = cameraAtIndex(cameraIdx);
    if (c.haveIntrinsicCalibration()) {
      updateCameraIntrinsicDistortionParameters(cameraIdx);
      printf("CameraSystem::loadCalibrationData: Camera %zu: Loaded intrinsic calibration\n", cameraIdx);
    } else {
      printf("CameraSystem::loadCalibrationData: Camera %zu: Intrinsic calibration required\n", cameraIdx);
    }
  }

  for (size_t viewIdx = 0; viewIdx < m_views.size(); ++viewIdx) {
    View& v = viewAtIndex(viewIdx);
    if (v.isStereo) {
      if (v.haveStereoCalibration()) {
        updateViewStereoDistortionParameters(viewIdx);
        printf("CameraSystem::loadCalibrationData: View %zu: Loaded stereo calibration\n", viewIdx);
      } else {
        printf("CameraSystem::loadCalibrationData: View %zu: Stereo calibration required\n", viewIdx);
      }
    }
  }

  return true;
}

void CameraSystem::saveCalibrationData() {
  cv::FileStorage fs(calibrationFilename.c_str(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);

  fs.startWriteStruct("cameras", cv::FileNode::SEQ);
  for (size_t cameraIdx = 0; cameraIdx < m_cameras.size(); ++cameraIdx) {
    fs.startWriteStruct(cv::String(), cv::FileNode::MAP);
    Camera& c = cameraAtIndex(cameraIdx);
    if (c.haveIntrinsicCalibration()) {
      fs.write("intrinsicMatrix", c.intrinsicMatrix);
      fs.write("distortionCoeffs", c.distCoeffs);
    }
    fs.endWriteStruct();
  }
  fs.endWriteStruct();

  fs.startWriteStruct("views", cv::FileNode::SEQ);
  for (size_t viewIdx = 0; viewIdx < m_views.size(); ++viewIdx) {
    fs.startWriteStruct(cv::String(), cv::FileNode::MAP);
    View& v = viewAtIndex(viewIdx);
    fs.write("isStereo", v.isStereo);
    if (v.isStereo) {
      fs.write("leftCameraIndex", (int) v.cameraIndices[0]);
      fs.write("rightCameraIndex", (int) v.cameraIndices[1]);
      if (v.haveStereoCalibration()) {
        fs.write("stereoRotation", v.stereoRotation);
        fs.write("stereoTranslation", v.stereoTranslation);
      }
    } else {
      fs.write("cameraIndex", (int) v.cameraIndices[0]);
    }
    
    fs.endWriteStruct();
  }

}

void CameraSystem::updateCameraIntrinsicDistortionParameters(size_t cameraIdx) {
  cv::Size imageSize = cv::Size(argusCamera()->streamWidth(), argusCamera()->streamHeight());
  float alpha = 0.25; // scaling factor. 0 = no invalid pixels in output (no black borders), 1 = use all input pixels
  cv::Mat map1, map2;
  Camera& c = cameraAtIndex(cameraIdx);

  // Intrinic-only remap -- no rectification transform, using optimized matrix
  c.optimizedMatrix = cv::getOptimalNewCameraMatrix(c.intrinsicMatrix, c.distCoeffs, imageSize, alpha, cv::Size(), NULL, /*centerPrincipalPoint=*/true);
  cv::initUndistortRectifyMap(c.intrinsicMatrix, c.distCoeffs, cv::noArray(), c.optimizedMatrix, imageSize, CV_32F, map1, map2);

  // Compute FOV
  {
    double focalLength, aspectRatio; // not valid without a real aperture size, which we don't bother providing
    cv::Point2d principalPoint;

    cv::calibrationMatrixValues(c.optimizedMatrix, cv::Size(argusCamera()->streamWidth(), argusCamera()->streamHeight()), 0.0, 0.0, c.fovX, c.fovY, focalLength, principalPoint, aspectRatio);
  }
  
  c.intrinsicDistortionMap = generateGPUDistortionMap(map1, map2);
}

void CameraSystem::updateViewStereoDistortionParameters(size_t viewIdx) {
  cv::Size imageSize = cv::Size(argusCamera()->streamWidth(), argusCamera()->streamHeight());
  cv::Mat map1, map2;
  View& v = viewAtIndex(viewIdx);
  Camera& leftC = cameraAtIndex(v.cameraIndices[0]);
  Camera& rightC = cameraAtIndex(v.cameraIndices[1]);

  // Compute rectification/projection transforms from the stereo calibration data
  float alpha = -1.0f;  //0.25;

  // Using the optimized camera matrix and zero distortion matrix again to create a rectification remap that can be layered on top of the intrinsic distortion remap
  cv::stereoRectify(
    leftC.optimizedMatrix, zeroDistortion,
    rightC.optimizedMatrix, zeroDistortion,
    cv::Size(argusCamera()->streamWidth(), argusCamera()->streamHeight()),
    v.stereoRotation, v.stereoTranslation,
    v.stereoRectification[0], v.stereoRectification[1],
    v.stereoProjection[0], v.stereoProjection[1],
    v.stereoDisparityToDepth,
    /*flags=*/cv::CALIB_ZERO_DISPARITY, alpha, cv::Size(),
    &v.stereoValidROI[0], &v.stereoValidROI[1]);

  // Camera info dump
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    printf("\n ===== View %zu | %s Camera (%u) ===== \n", viewIdx, eyeIdx == 0 ? "Left" : "Right", v.cameraIndices[eyeIdx]);

    std::cout << "* Stereo Rectification matrix:" << std::endl << v.stereoRectification[eyeIdx] << std::endl;
    std::cout << "* Stereo Projection matrix:" << std::endl << v.stereoProjection[eyeIdx] << std::endl;
    std::cout << "* Stereo Valid ROI:" << std::endl << v.stereoValidROI[eyeIdx] << std::endl;


    const double apertureSize = 6.35; // mm, 1/4" sensor. TODO parameterize
    double fovX, fovY, focalLength, aspectRatio;
    cv::Point2d principalPoint;

    Camera& c = cameraAtIndex(v.cameraIndices[eyeIdx]);
    cv::calibrationMatrixValues(c.intrinsicMatrix, cv::Size(argusCamera()->streamWidth(), argusCamera()->streamHeight()), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
    printf("* Intrinsic matrix: FOV %.1f x %.1f deg, approx focal length %.2fmm\n", fovX, fovY, focalLength);
    cv::calibrationMatrixValues(c.optimizedMatrix, cv::Size(argusCamera()->streamWidth(), argusCamera()->streamHeight()), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
    printf("* Optimized matrix: FOV %.1f x %.1f deg\n", fovX, fovY);
    cv::calibrationMatrixValues(cv::Mat(v.stereoProjection[eyeIdx], cv::Rect(0, 0, 3, 3)), cv::Size(argusCamera()->streamWidth(), argusCamera()->streamHeight()), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
    printf("* Stereo projection matrix: FOV %.1f x %.1f deg\n", fovX, fovY);
  }
  printf("\n ==================== \n");
  std::cout << "View " << viewIdx << " Stereo translation:" << std::endl << v.stereoTranslation << std::endl;
  std::cout << "View " << viewIdx << " Stereo rotation matrix:" << std::endl << v.stereoRotation << std::endl;

  {
    // Compute FOV. Should be the same for stereoProjection[0] and [1], so we only keep a single value.
    double focalLength, aspectRatio;
    cv::Point2d principalPoint;

    cv::calibrationMatrixValues(cv::Mat(v.stereoProjection[0], cv::Rect(0, 0, 3, 3)), cv::Size(argusCamera()->streamWidth(), argusCamera()->streamHeight()), 0.0, 0.0, v.fovX, v.fovY, focalLength, principalPoint, aspectRatio);
  }

  // Check the valid image regions for a failed stereo calibration. A bad calibration will usually result in a valid ROI for one or both views with a 0-pixel dimension.
/*
  if (needStereoCalibration) {
    if (stereoValidROI[0].area() == 0 || stereoValidROI[1].area() == 0) {
      printf("Stereo calibration failed: one or both of the valid image regions has zero area.\n");
      goto retryStereoCalibration;
    }
  }
*/

  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    Camera& c = cameraAtIndex(v.cameraIndices[eyeIdx]);

    cv::initUndistortRectifyMap(c.intrinsicMatrix, c.distCoeffs, v.stereoRectification[eyeIdx], v.stereoProjection[eyeIdx], imageSize, CV_32F, map1, map2);
    
    v.stereoDistortionMap[eyeIdx] = generateGPUDistortionMap(map1, map2);
  }
}

RHISurface::ptr CameraSystem::generateGPUDistortionMap(cv::Mat map1, cv::Mat map2) {
  assert(map1.rows == map2.rows && map1.cols == map2.cols);
  size_t width = map1.cols;
  size_t height = map1.rows;

  // map1 and map2 should contain absolute x and y coords for sampling the input image, in pixel scale (map1 is 0-1280, map2 is 0-720, etc) -- this is the output format of cv::initUndistortRectifyMap
  // Combine the maps into a buffer we can upload to opengl. Remap the absolute pixel coordinates to UV (0...1) range to save work in the pixel shader.
  float* distortionMapTmp = new float[width * height * 2];
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      // .at(row, col) -- Y rows, X columns.
      distortionMapTmp[(((y * width) + x) * 2) + 0] = map1.at<float>(y, x) / static_cast<float>(width);
      distortionMapTmp[(((y * width) + x) * 2) + 1] = map2.at<float>(y, x) / static_cast<float>(height);
    }
  }

  RHISurface::ptr distortionMap = rhi()->newTexture2D(width, height, RHISurfaceDescriptor(kSurfaceFormat_RG32f));
  rhi()->loadTextureData(distortionMap, kVertexElementTypeFloat2, distortionMapTmp);

  delete[] distortionMapTmp;
  return distortionMap;
}

size_t CameraSystem::createMonoView(size_t cameraIndex) {
  assert(cameraIndex < m_cameras.size());

  m_views.resize(m_views.size() + 1);
  size_t newViewIndex = m_views.size() - 1;

  View& v = m_views[newViewIndex];

  v.isStereo = false;
  v.cameraIndices[0] = cameraIndex;
  v.cameraIndices[1] = cameraIndex;

  return newViewIndex;
}

size_t CameraSystem::createStereoView(size_t leftCameraIndex, size_t rightCameraIndex) {
  assert(leftCameraIndex < m_cameras.size());
  assert(rightCameraIndex < m_cameras.size());

  m_views.resize(m_views.size() + 1);
  size_t newViewIndex = m_views.size() - 1;

  View& v = m_views[newViewIndex];

  v.isStereo = true;
  v.cameraIndices[0] = leftCameraIndex;
  v.cameraIndices[1] = rightCameraIndex;

  return newViewIndex;
}


CameraSystem::CalibrationContext* CameraSystem::calibrationContextForCamera(size_t cameraIdx) {
  assert(cameraIdx < cameras());
  return new CameraSystem::IntrinsicCalibrationContext(this, cameraIdx);
}
CameraSystem::CalibrationContext* CameraSystem::calibrationContextForView(size_t viewIdx) {
  assert(viewIdx < views() && viewAtIndex(viewIdx).isStereo);
  return new CameraSystem::StereoCalibrationContext(this, viewIdx);
}


CameraSystem::CalibrationContext::CalibrationContext(CameraSystem* cs) : m_cameraSystem(cs) {

}

CameraSystem::CalibrationContext::~CalibrationContext() {

}

CameraSystem::CalibrationContextStateMachineBase::CalibrationContextStateMachineBase(CameraSystem* cs) : CameraSystem::CalibrationContext(cs), 
  m_captureRequested(false), m_finishRequested(false), m_cancelRequested(false), m_saveCalibrationImages(false), m_inCalibrationPreviewMode(false), m_calibrationPreviewAccepted(false), m_calibrationPreviewRejected(false), m_calibrationFinished(false) {

}

CameraSystem::CalibrationContextStateMachineBase::~CalibrationContextStateMachineBase() {

}


bool CameraSystem::CalibrationContextStateMachineBase::finished() {
  return m_calibrationFinished;
}


void CameraSystem::CalibrationContextStateMachineBase::processUI() {
  if (finished())
    return;

  this->renderStatusUI();

  if (m_inCalibrationPreviewMode) {
    m_calibrationPreviewAccepted = ImGui::Button("Accept Calibration");
    m_calibrationPreviewRejected = ImGui::Button("Reject Calibration");
  } else {
    m_captureRequested = ImGui::Button("Capture Frame");
    ImGui::Checkbox("Save calibration images", &m_saveCalibrationImages);
    m_finishRequested = ImGui::Button("Finish Calibration");
    m_cancelRequested = ImGui::Button("Cancel and Discard");
  }

}

cv::Mat CameraSystem::captureGreyscale(size_t cameraIdx, RHISurface::ptr tex, RHIRenderTarget::ptr rt, bool undistort) {

  rhi()->beginRenderPass(rt, kLoadInvalidate);
  if (undistort) {
    assert(cameraAtIndex(cameraIdx).intrinsicDistortionMap);

    rhi()->bindRenderPipeline(camGreyscaleUndistortPipeline);
    rhi()->loadTexture(ksDistortionMap, cameraAtIndex(cameraIdx).intrinsicDistortionMap, linearClampSampler);
  } else {
    rhi()->bindRenderPipeline(camGreyscalePipeline);
  }
  rhi()->loadTexture(ksImageTex, m_argusCamera->rgbTexture(cameraIdx), linearClampSampler);
  rhi()->drawNDCQuad();
  rhi()->endRenderPass(rt);

  cv::Mat res;
  res.create(/*rows=*/ tex->height(), /*columns=*/tex->width(), CV_8UC1);
  assert(res.isContinuous());
  rhi()->readbackTexture(tex, 0, kVertexElementTypeUByte1N, res.ptr(0));
  return res;
}


















CameraSystem::IntrinsicCalibrationContext::IntrinsicCalibrationContext(CameraSystem* cs, size_t cameraIdx) : CameraSystem::CalibrationContextStateMachineBase(cs), m_cameraIdx(cameraIdx) {

  // Store cancellation cache
  m_previousIntrinsicMatrix = cameraSystem()->cameraAtIndex(cameraIdx).intrinsicMatrix;
  m_previousDistCoeffs = cameraSystem()->cameraAtIndex(cameraIdx).distCoeffs;

  // Textures and RTs we use for captures
  m_fullGreyTex = rhi()->newTexture2D(argusCamera()->streamWidth(), argusCamera()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));
  m_fullGreyRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({m_fullGreyTex}));
  m_feedbackTex = rhi()->newTexture2D(argusCamera()->streamWidth(), argusCamera()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
  m_feedbackView.create(/*rows=*/ argusCamera()->streamHeight(), /*columns=*/argusCamera()->streamWidth(), CV_8UC4);
}

CameraSystem::IntrinsicCalibrationContext::~IntrinsicCalibrationContext() {

}

void CameraSystem::IntrinsicCalibrationContext::renderStatusUI() {
  ImGui::Text("Camera %zu Intrinsic", m_cameraIdx);
  ImGui::Text("%zu samples", m_allCharucoCorners.size());
}

void CameraSystem::CalibrationContextStateMachineBase::processFrame() {

  if (m_calibrationFinished) {
    return; // Nothing to do
  }

  if (!m_inCalibrationPreviewMode) {
    this->processFrameCaptureMode();
  }

  if (m_finishRequested) {
    m_finishRequested = false;
    m_captureRequested = false; // Skip any stale capture requests

    // Cook a new calibration from the captures and switch to preview mode

    m_calibrationPreviewAccepted = false;
    m_calibrationPreviewRejected = false;

    if (this->cookCalibrationDataForPreview()) {
      // Calibration succeeded, enter preview mode

      m_inCalibrationPreviewMode = true;

    } else {
      // Calibration failed, back to capture mode
      m_calibrationPreviewRejected = true;
    }
  }


  if (m_inCalibrationPreviewMode) {
    if (m_calibrationPreviewAccepted) {
      m_calibrationPreviewAccepted = false;
      // Mark the context as finished
      this->didAcceptCalibrationPreview();

      // Save the updated calibration data
      cameraSystem()->saveCalibrationData();

      m_calibrationFinished = true;
    }

    else if (m_calibrationPreviewRejected) {
      m_calibrationPreviewRejected = false;
      // Reset stored data and return to capture mode
      this->didRejectCalibrationPreview();

      m_inCalibrationPreviewMode = false;
    }

  }

  if (m_cancelRequested) {
    m_cancelRequested = false;
    this->didCancelCalibrationSession();
    m_calibrationFinished = true;
  }
}

void CameraSystem::IntrinsicCalibrationContext::didAcceptCalibrationPreview() {
  // Calibration accepted. We don't need to do anything else here.
}

void CameraSystem::IntrinsicCalibrationContext::didRejectCalibrationPreview() {
  // Reset stored data in preparation for reentering capture mode
  m_allCharucoCorners.clear();
  m_allCharucoIds.clear();
}

void CameraSystem::IntrinsicCalibrationContext::didCancelCalibrationSession() {
  // Restore previously saved calibration snapshot
  cameraSystem()->cameraAtIndex(m_cameraIdx).intrinsicMatrix = m_previousIntrinsicMatrix;
  cameraSystem()->cameraAtIndex(m_cameraIdx).distCoeffs = m_previousDistCoeffs;
}

bool CameraSystem::IntrinsicCalibrationContext::cookCalibrationDataForPreview() {
  try {
    Camera& c = cameraSystem()->cameraAtIndex(m_cameraIdx);

    cv::Mat stdDeviations, perViewErrors;
    std::vector<float> reprojErrs;
    cv::Size imageSize(argusCamera()->streamWidth(), argusCamera()->streamHeight());
    float aspectRatio = 1.0f;
    int flags = cv::CALIB_FIX_PRINCIPAL_POINT | cv::CALIB_FIX_ASPECT_RATIO;

    c.intrinsicMatrix = cv::Mat::eye(3, 3, CV_64F);
    if( flags & cv::CALIB_FIX_ASPECT_RATIO )
        c.intrinsicMatrix.at<double>(0,0) = aspectRatio;
    c.distCoeffs = cv::Mat::zeros(8, 1, CV_64F);


    double rms = cv::aruco::calibrateCameraCharuco(m_allCharucoCorners, m_allCharucoIds,
                                   s_charucoBoard, imageSize,
                                   c.intrinsicMatrix, c.distCoeffs,
                                   cv::noArray(), cv::noArray(), stdDeviations, cv::noArray(),
                                   perViewErrors, flags);


    printf("RMS error reported by calibrateCameraCharuco: %g\n", rms);
    std::cout << "Camera " << m_cameraIdx << " Per-view error: " << std::endl << perViewErrors << std::endl;
    std::cout << "Camera " << m_cameraIdx << " Matrix: " << std::endl << c.intrinsicMatrix << std::endl;
    std::cout << "Camera " << m_cameraIdx << " Distortion coefficients: " << std::endl << c.distCoeffs << std::endl;

    // Recompute distortion parameters and update map
    cameraSystem()->updateCameraIntrinsicDistortionParameters(m_cameraIdx);


    return true; // Calibration succeeded

  } catch (const std::exception& ex) {
    printf("Camera intrinsic calibration failed: %s\n", ex.what());
  }
  return false; // Calibration failed
}

void CameraSystem::IntrinsicCalibrationContext::processFrameCaptureMode() {
  // Look for ChAruCo markers and allow capturing if we're not in preview mode

  // stereoCamera->readFrame();
  cv::Mat viewFullRes = cameraSystem()->captureGreyscale(m_cameraIdx, m_fullGreyTex, m_fullGreyRT, /*undistort=*/false);

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
  memset(m_feedbackView.ptr(0), 0, m_feedbackView.total() * 4);

  if (!ids.empty()) {
    cv::aruco::drawDetectedMarkers(m_feedbackView, corners);
  }
  if (currentCharucoCorners.total() > 3) {
    cv::aruco::drawDetectedCornersCharuco(m_feedbackView, currentCharucoCorners, currentCharucoIds);
  }

  rhi()->loadTextureData(m_feedbackTex, kVertexElementTypeUByte4N, m_feedbackView.ptr(0));

  // Require at least a third of the markers to be in frame to take an intrinsic calibration sample
  bool found = (currentCharucoCorners.total() >= (s_charucoBoard->chessboardCorners.size() / 3));
  if (found && captureRequested()) {
    acknowledgeCaptureRequest();

    m_allCharucoCorners.push_back(currentCharucoCorners);
    m_allCharucoIds.push_back(currentCharucoIds);

    if (shouldSaveCalibrationImages()) {
      char filename[128];
      sprintf(filename, "calib_%zu_%02zu_overlay.png", m_cameraIdx, m_allCharucoCorners.size());

      // composite with the greyscale view and fix the alpha channel before writing
      for (size_t pixelIdx = 0; pixelIdx < (argusCamera()->streamWidth() * argusCamera()->streamHeight()); ++pixelIdx) {
        uint8_t* p = m_feedbackView.ptr(0) + (pixelIdx * 4);
        if (!(p[0] || p[1] || p[2])) {
          p[0] = p[1] = p[2] = viewFullRes.ptr(0)[pixelIdx];

        }
        p[3] = 0xff;
      }
      stbi_write_png(filename, argusCamera()->streamWidth(), argusCamera()->streamHeight(), 4, m_feedbackView.ptr(0), /*rowBytes=*/argusCamera()->streamWidth() * 4);
      printf("Saved %s\n", filename);
    }

  }
}

bool CameraSystem::IntrinsicCalibrationContext::requiresStereoRendering() const {
  return false;
}

RHISurface::ptr CameraSystem::IntrinsicCalibrationContext::overlaySurfaceAtIndex(size_t) {
  return m_feedbackTex;
}







CameraSystem::StereoCalibrationContext::StereoCalibrationContext(CameraSystem* cs, size_t viewIdx) : CameraSystem::CalibrationContextStateMachineBase(cs), m_viewIdx(viewIdx) {

  // Store cancellation cache
  m_previousViewData = cameraSystem()->viewAtIndex(viewIdx);

  // Textures and RTs we use for captures
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    m_fullGreyTex[eyeIdx] = rhi()->newTexture2D(argusCamera()->streamWidth(), argusCamera()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));
    m_fullGreyRT[eyeIdx] = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({m_fullGreyTex[eyeIdx]}));
    m_feedbackTex[eyeIdx] = rhi()->newTexture2D(argusCamera()->streamWidth(), argusCamera()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    m_feedbackView[eyeIdx].create(/*rows=*/ argusCamera()->streamHeight(), /*columns=*/argusCamera()->streamWidth(), CV_8UC4);
  }
}

CameraSystem::StereoCalibrationContext::~StereoCalibrationContext() {

}

void CameraSystem::StereoCalibrationContext::renderStatusUI() {
  ImGui::Text("View %zu Stereo", m_viewIdx);
  ImGui::Text("%zu samples", m_objectPoints.size());
}


void CameraSystem::StereoCalibrationContext::processFrameCaptureMode() {
  // Capture and undistort camera views.
  cv::Mat eyeFullRes[2];
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    eyeFullRes[eyeIdx] = cameraSystem()->captureGreyscale(cameraSystem()->viewAtIndex(m_viewIdx).cameraIndices[eyeIdx], m_fullGreyTex[eyeIdx], m_fullGreyRT[eyeIdx], /*undistort=*/true);
  }

  std::vector<std::vector<cv::Point2f> > corners[2], rejected[2];
  std::vector<int> ids[2];

  std::vector<cv::Point2f> currentCharucoCornerPoints[2];
  std::vector<int> currentCharucoCornerIds[2];

  // Run ArUco marker detection
  // Note that we don't feed the camera distortion parameters to the aruco functions here, since the images we're operating on have already been undistorted.
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    Camera& c = cameraSystem()->cameraAtIndex(cameraSystem()->viewAtIndex(m_viewIdx).cameraIndices[eyeIdx]);

    cv::aruco::detectMarkers(eyeFullRes[eyeIdx], s_charucoDictionary, corners[eyeIdx], ids[eyeIdx], cv::aruco::DetectorParameters::create(), rejected[eyeIdx], c.optimizedMatrix, zeroDistortion);
    cv::aruco::refineDetectedMarkers(eyeFullRes[eyeIdx], s_charucoBoard, corners[eyeIdx], ids[eyeIdx], rejected[eyeIdx], c.optimizedMatrix, zeroDistortion);

    // Find chessboard corners using detected markers
    if (!ids[eyeIdx].empty()) {
      cv::aruco::interpolateCornersCharuco(corners[eyeIdx], ids[eyeIdx], eyeFullRes[eyeIdx], s_charucoBoard, currentCharucoCornerPoints[eyeIdx], currentCharucoCornerIds[eyeIdx], c.optimizedMatrix, zeroDistortion);
    }
  }

  // Find set of chessboard corners present in both eyes
  std::set<int> stereoCornerIds;
  {
    std::set<int> eye0Ids;
    for (size_t i = 0; i < currentCharucoCornerIds[0].size(); ++i) {
      eye0Ids.insert(currentCharucoCornerIds[0][i]);
    }
    for (size_t i = 0; i < currentCharucoCornerIds[1].size(); ++i) {
      int id = currentCharucoCornerIds[1][i];
      if (eye0Ids.find(id) != eye0Ids.end())
        stereoCornerIds.insert(id);
    }
  }

  // Require at least 6 corners visibile to both cameras to consider this frame
  bool foundOverlap = stereoCornerIds.size() >= 6;

  // Filter the eye corner sets to only overlapping corners, which we will later feed to stereoCalibrate
  std::vector<cv::Point3f> thisFrameBoardRefCorners;
  std::vector<cv::Point2f> thisFrameImageCorners[2];

  for (std::set<int>::const_iterator corner_it = stereoCornerIds.begin(); corner_it != stereoCornerIds.end(); ++corner_it) {
    int cornerId = *corner_it;

    for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
      for (size_t eyeCornerIdx = 0; eyeCornerIdx < currentCharucoCornerIds[eyeIdx].size(); ++eyeCornerIdx) {
        if (currentCharucoCornerIds[eyeIdx][eyeCornerIdx] == cornerId) {
          thisFrameImageCorners[eyeIdx].push_back(currentCharucoCornerPoints[eyeIdx][eyeCornerIdx]);
          break;
        }
      }
    }

    // save the corner point in board space from the board definition
    thisFrameBoardRefCorners.push_back(s_charucoBoard->chessboardCorners[cornerId]);
  }
  assert(thisFrameBoardRefCorners.size() == thisFrameImageCorners[0].size() && thisFrameBoardRefCorners.size() == thisFrameImageCorners[1].size());


  // Draw feedback points
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    memset(m_feedbackView[eyeIdx].ptr(0), 0, m_feedbackView[eyeIdx].total() * 4);

    if (!corners[eyeIdx].empty()) {
      cv::aruco::drawDetectedMarkers(m_feedbackView[eyeIdx], corners[eyeIdx]);
    }

    // Borrowed from cv::aruco::drawDetectedCornersCharuco -- modified to switch the color per-marker to indicate stereo visibility
    for(size_t cornerIdx = 0; cornerIdx < currentCharucoCornerIds[eyeIdx].size(); ++cornerIdx) {
      cv::Point2f corner = currentCharucoCornerPoints[eyeIdx][cornerIdx];
      int id = currentCharucoCornerIds[eyeIdx][cornerIdx];

      // grey for mono points
      cv::Scalar cornerColor = cv::Scalar(127, 127, 127);
      if (stereoCornerIds.find(id) != stereoCornerIds.end()) {
        // red for stereo points
        cornerColor = cv::Scalar(255, 0, 0);
      }

      // draw first corner mark
      cv::rectangle(m_feedbackView[eyeIdx], corner - cv::Point2f(3, 3), corner + cv::Point2f(3, 3), cornerColor, 1, cv::LINE_AA);

      // draw ID
      char idbuf[16];
      sprintf(idbuf, "id=%u", id);
      cv::putText(m_feedbackView[eyeIdx], idbuf, corner + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cornerColor, 2);
    }
  }
  rhi()->loadTextureData(m_feedbackTex[0], kVertexElementTypeUByte4N, m_feedbackView[0].ptr(0));
  rhi()->loadTextureData(m_feedbackTex[1], kVertexElementTypeUByte4N, m_feedbackView[1].ptr(0));


  // Handle capture requests
  if (foundOverlap && captureRequested()) {
    acknowledgeCaptureRequest();

    m_objectPoints.push_back(thisFrameBoardRefCorners);
    m_calibrationPoints[0].push_back(thisFrameImageCorners[0]);
    m_calibrationPoints[1].push_back(thisFrameImageCorners[1]);

#if 0 // TODO fix this
    if (shouldSaveCalibrationImages()) {
      char filename[128];
      sprintf(filename, "calib_view_%zu_%02u_overlay.png", m_viewIdx, m_objectPoints.size());

      // composite with the greyscale view and fix the alpha channel before writing
      for (size_t pixelIdx = 0; pixelIdx < (argusCamera()->streamWidth() * 2) * argusCamera()->streamHeight(); ++pixelIdx) {
        uint8_t* p = feedbackViewStereo.ptr(0) + (pixelIdx * 4);
        if (!(p[0] || p[1] || p[2])) {
          p[0] = p[1] = p[2] = viewFullResStereo.ptr(0)[pixelIdx];

        }
        p[3] = 0xff;
      }
      stbi_write_png(filename, argusCamera()->streamWidth()*2, argusCamera()->streamHeight(), 4, feedbackViewStereo.ptr(0), /*rowBytes=*/argusCamera()->streamWidth()*2 * 4);
      printf("Saved %s\n", filename);
    }
#endif

  }
}


bool CameraSystem::StereoCalibrationContext::cookCalibrationDataForPreview() {
  try {
    // Samples collected, run calibration
    cv::Mat E, F;
    cv::Mat perViewErrors;

    // Note the use of the zero distortion matrix and optimized camera matrix, since we already corrected for individual camera distortions when capturing the images.
    View& v = cameraSystem()->viewAtIndex(m_viewIdx);

    double rms = cv::stereoCalibrate(m_objectPoints,
      m_calibrationPoints[0], m_calibrationPoints[1],
      cameraSystem()->cameraAtIndex(v.cameraIndices[0]).optimizedMatrix, zeroDistortion,
      cameraSystem()->cameraAtIndex(v.cameraIndices[1]).optimizedMatrix, zeroDistortion,
      cv::Size(argusCamera()->streamWidth(), argusCamera()->streamHeight()),
      v.stereoRotation, v.stereoTranslation, E, F, perViewErrors, cv::CALIB_FIX_INTRINSIC);

    printf("RMS error reported by stereoCalibrate: %g\n", rms);
    std::cout << " Per-view error: " << std::endl << perViewErrors << std::endl;

    cameraSystem()->updateViewStereoDistortionParameters(m_viewIdx);

    return true;
  } catch (const std::exception& ex) {
    printf("Stereo calibration failed: %s\n", ex.what());
  }
  return false;
}

void CameraSystem::StereoCalibrationContext::didAcceptCalibrationPreview() {
  // No action required
}

void CameraSystem::StereoCalibrationContext::didRejectCalibrationPreview() {
  // Reset stored data
  m_objectPoints.clear();
  m_calibrationPoints[0].clear();
  m_calibrationPoints[1].clear();
}

void CameraSystem::StereoCalibrationContext::didCancelCalibrationSession() {
  // Restore previously saved calibration snapshot
  cameraSystem()->viewAtIndex(m_viewIdx) = m_previousViewData;

  if (cameraSystem()->viewAtIndex(m_viewIdx).haveStereoCalibration())
    cameraSystem()->updateViewStereoDistortionParameters(m_viewIdx);
}

bool CameraSystem::StereoCalibrationContext::requiresStereoRendering() const {
  return true;
}

RHISurface::ptr CameraSystem::StereoCalibrationContext::overlaySurfaceAtIndex(size_t index) {
  assert(index < 2);
  return m_feedbackTex[index];
}
