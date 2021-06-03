#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/CharucoMultiViewCalibration.h"
#include "common/glmCvInterop.h"
#include "common/FxThreading.h"
#include "rhi/RHIResources.h"
#include "imgui.h"
#include <iostream>
#include <set>
#include <assert.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

static const cv::Mat zeroDistortion = cv::Mat::zeros(1, 14, CV_64F);

// ChAruCo target pattern config
const cv::aruco::PREDEFINED_DICTIONARY_NAME s_charucoDictionaryName = cv::aruco::DICT_5X5_100;
const unsigned int s_charucoBoardSquareCountX = 12;
const unsigned int s_charucoBoardSquareCountY = 9;
const float s_charucoBoardSquareSideLengthMeters = 0.060f;
// markers are 7x7 pixels, squares are 9x9 pixels (add 1px border), so the marker size is 7/9 of the square size
const float s_charucoBoardMarkerSideLengthMeters = s_charucoBoardSquareSideLengthMeters * (7.0f / 9.0f);

cv::Ptr<cv::aruco::Dictionary> s_charucoDictionary;
cv::Ptr<cv::aruco::CharucoBoard> s_charucoBoard;


// TODO move this
RHIRenderPipeline::ptr camGreyscalePipeline;
RHIRenderPipeline::ptr camGreyscaleUndistortPipeline;
FxAtomicString ksDistortionMap("distortionMap");
FxAtomicString ksImageTex("imageTex");


CameraSystem::CameraSystem(ICameraProvider* cam) : calibrationFilename("calibration.yml"), m_cameraProvider(cam) {

  // Initialize ChAruCo data on first use
  if (!s_charucoDictionary)
    s_charucoDictionary = cv::aruco::getPredefinedDictionary(s_charucoDictionaryName);

  if (!s_charucoBoard)
    s_charucoBoard = cv::aruco::CharucoBoard::create(s_charucoBoardSquareCountX, s_charucoBoardSquareCountY, s_charucoBoardSquareSideLengthMeters, s_charucoBoardMarkerSideLengthMeters, s_charucoDictionary);

  m_cameras.resize(cameraProvider()->streamCount());


  // Compile utility pipelines on first use
  if (!camGreyscalePipeline) {
    RHIShaderDescriptor desc(
      "shaders/ndcQuad.vtx.glsl",
      "shaders/camGreyscale.frag.glsl",
      ndcQuadVertexLayout);
#ifdef GLATTER_EGL_GLES_3_2 // TODO query this at use-time from the RHISurface type
    desc.setFlag("SAMPLER_TYPE", "samplerExternalOES");
#else
    desc.setFlag("SAMPLER_TYPE", "sampler2D");
#endif
    camGreyscalePipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }

  if (!camGreyscaleUndistortPipeline) {
    RHIShaderDescriptor desc(
      "shaders/ndcQuad.vtx.glsl",
      "shaders/camGreyscaleUndistort.frag.glsl",
      ndcQuadVertexLayout);
#ifdef GLATTER_EGL_GLES_3_2
    desc.setFlag("SAMPLER_TYPE", "samplerExternalOES");
#else
    desc.setFlag("SAMPLER_TYPE", "sampler2D");
#endif
    camGreyscaleUndistortPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), tristripPipelineDescriptor);
  }


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
            vfn["essentialMatrix"] >> v.essentialMatrix;
            vfn["fundamentalMatrix"] >> v.fundamentalMatrix;

            cv::Mat tmpMat;
            vfn["viewTranslation"] >> tmpMat; v.viewTranslation = glmVec3FromCV(tmpMat);
            vfn["viewRotation"] >> tmpMat; v.viewRotation = glmVec3FromCV(tmpMat);
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

  fs.startWriteStruct("cameras", cv::FileNode::SEQ, cv::String());
  for (size_t cameraIdx = 0; cameraIdx < m_cameras.size(); ++cameraIdx) {
    fs.startWriteStruct(cv::String(), cv::FileNode::MAP, cv::String());
    Camera& c = cameraAtIndex(cameraIdx);
    if (c.haveIntrinsicCalibration()) {
      fs.write("intrinsicMatrix", c.intrinsicMatrix);
      fs.write("distortionCoeffs", c.distCoeffs);
    }
    fs.endWriteStruct();
  }
  fs.endWriteStruct();

  fs.startWriteStruct("views", cv::FileNode::SEQ, cv::String());
  for (size_t viewIdx = 0; viewIdx < m_views.size(); ++viewIdx) {
    fs.startWriteStruct(cv::String(), cv::FileNode::MAP, cv::String());
    View& v = viewAtIndex(viewIdx);
    fs.write("isStereo", v.isStereo);
    if (v.isStereo) {
      fs.write("leftCameraIndex", (int) v.cameraIndices[0]);
      fs.write("rightCameraIndex", (int) v.cameraIndices[1]);
      if (v.haveStereoCalibration()) {
        fs.write("stereoRotation", v.stereoRotation);
        fs.write("stereoTranslation", v.stereoTranslation);
        fs.write("essentialMatrix", v.essentialMatrix);
        fs.write("fundamentalMatrix", v.fundamentalMatrix);
      }
      fs.write("viewTranslation", cv::Mat(cvVec3FromGlm(v.viewTranslation)));
      fs.write("viewRotation", cv::Mat(cvVec3FromGlm(v.viewRotation)));
    } else {
      fs.write("cameraIndex", (int) v.cameraIndices[0]);
    }
    
    fs.endWriteStruct();
  }

}

glm::mat4 CameraSystem::viewWorldTransform(size_t viewIdx) const {
  assert(viewIdx < views());

  glm::mat4 vt = viewAtIndex(viewIdx).viewLocalTransform();

  // View 0's tranform is stored relative to the world (user input)
  // Other view tranforms are calibrated, so they're stored relative to view 0 to maintain calibration when adjusting the view offset.

  if (viewIdx == 0)
    return vt;
  else
    return viewAtIndex(0).viewLocalTransform() * vt;

}

void CameraSystem::updateCameraIntrinsicDistortionParameters(size_t cameraIdx) {
  cv::Size imageSize = cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight());
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

    cv::calibrationMatrixValues(c.optimizedMatrix, cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()), 0.0, 0.0, c.fovX, c.fovY, focalLength, principalPoint, aspectRatio);
  }
  
  c.intrinsicDistortionMap = generateGPUDistortionMap(map1, map2);
}

void CameraSystem::updateViewStereoDistortionParameters(size_t viewIdx) {
  cv::Size imageSize = cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight());
  cv::Mat map1, map2;
  View& v = viewAtIndex(viewIdx);
  Camera& leftC = cameraAtIndex(v.cameraIndices[0]);
  Camera& rightC = cameraAtIndex(v.cameraIndices[1]);

  // Compute rectification/projection transforms from the stereo calibration data
  float alpha = -1.0f;  //0.25;

  cv::stereoRectify(
    leftC.intrinsicMatrix, leftC.distCoeffs,
    rightC.intrinsicMatrix, rightC.distCoeffs,
    cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()),
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
    cv::calibrationMatrixValues(c.intrinsicMatrix, cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
    printf("* Intrinsic matrix: FOV %.1f x %.1f deg, approx focal length %.2fmm\n", fovX, fovY, focalLength);
    cv::calibrationMatrixValues(c.optimizedMatrix, cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
    printf("* Optimized matrix: FOV %.1f x %.1f deg\n", fovX, fovY);
    cv::calibrationMatrixValues(cv::Mat(v.stereoProjection[eyeIdx], cv::Rect(0, 0, 3, 3)), cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()), apertureSize, apertureSize, fovX, fovY, focalLength, principalPoint, aspectRatio);
    printf("* Stereo projection matrix: FOV %.1f x %.1f deg\n", fovX, fovY);
  }
  printf("\n ==================== \n");
  std::cout << "View " << viewIdx << " Stereo translation:" << std::endl << v.stereoTranslation << std::endl;
  std::cout << "View " << viewIdx << " Stereo rotation matrix:" << std::endl << v.stereoRotation << std::endl;

  {
    // Compute FOV. Should be the same for stereoProjection[0] and [1], so we only keep a single value.
    double focalLength, aspectRatio;
    cv::Point2d principalPoint;

    cv::calibrationMatrixValues(cv::Mat(v.stereoProjection[0], cv::Rect(0, 0, 3, 3)), cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()), 0.0, 0.0, v.fovX, v.fovY, focalLength, principalPoint, aspectRatio);
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

cv::Mat CameraSystem::captureGreyscale(size_t cameraIdx, RHISurface::ptr tex, RHIRenderTarget::ptr rt, RHISurface::ptr distortionMap) {

  rhi()->beginRenderPass(rt, kLoadInvalidate);
  if (distortionMap) {
    rhi()->bindRenderPipeline(camGreyscaleUndistortPipeline);
    rhi()->loadTexture(ksDistortionMap, distortionMap, linearClampSampler);
  } else {
    rhi()->bindRenderPipeline(camGreyscalePipeline);
  }
  rhi()->loadTexture(ksImageTex, cameraProvider()->rgbTexture(cameraIdx), linearClampSampler);
  rhi()->drawNDCQuad();
  rhi()->endRenderPass(rt);

  cv::Mat res;
  res.create(/*rows=*/ tex->height(), /*columns=*/tex->width(), CV_8UC1);
  assert(res.isContinuous());
  rhi()->readbackTexture(tex, 0, kVertexElementTypeUByte1N, res.ptr(0));
  return res;
}

CameraSystem::CalibrationContext* CameraSystem::calibrationContextForCamera(size_t cameraIdx) {
  assert(cameraIdx < cameras());
  return new CameraSystem::IntrinsicCalibrationContext(this, cameraIdx);
}
CameraSystem::CalibrationContext* CameraSystem::calibrationContextForView(size_t viewIdx) {
  assert(viewIdx < views() && viewAtIndex(viewIdx).isStereo);
  return new CameraSystem::StereoCalibrationContext(this, viewIdx);
}
CameraSystem::CalibrationContext* CameraSystem::calibrationContextForStereoViewOffset(size_t referenceViewIdx, size_t viewIdx) {
  assert(referenceViewIdx < views() && viewAtIndex(referenceViewIdx).isStereo);
  assert(viewIdx < views() && viewAtIndex(viewIdx).isStereo);
  return new CameraSystem::StereoViewOffsetCalibrationContext(this, referenceViewIdx, viewIdx);
}


CameraSystem::CalibrationContext::CalibrationContext(CameraSystem* cs) : m_cameraSystem(cs) {

}

CameraSystem::CalibrationContext::~CalibrationContext() {

}

CameraSystem::CalibrationContextStateMachineBase::CalibrationContextStateMachineBase(CameraSystem* cs) : CameraSystem::CalibrationContext(cs), 
  m_captureRequested(false), m_previewRequested(false), m_cancelRequested(false), m_saveCalibrationImages(false), m_inCalibrationPreviewMode(false), m_calibrationPreviewAccepted(false), m_calibrationPreviewRejected(false), m_calibrationFinished(false) {

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
    m_calibrationPreviewRejected = ImGui::Button("Back");
  } else {
    m_captureRequested = ImGui::Button("Capture Frame");
    ImGui::Checkbox("Save calibration images", &m_saveCalibrationImages);
    m_previewRequested = ImGui::Button("Preview Calibration");
    m_cancelRequested = ImGui::Button("Cancel and Discard");
  }

}
void CameraSystem::CalibrationContextStateMachineBase::processFrame() {

  if (m_calibrationFinished) {
    return; // Nothing to do
  }

  if (m_inCalibrationPreviewMode) {
    this->processFramePreviewMode();
  } else {
    this->processFrameCaptureMode();
  }

  if (m_previewRequested) {
    m_previewRequested = false;
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
      // Return to capture mode
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










CameraSystem::IntrinsicCalibrationContext::IntrinsicCalibrationContext(CameraSystem* cs, size_t cameraIdx) : CameraSystem::CalibrationContextStateMachineBase(cs), m_cameraIdx(cameraIdx) {

  // Store cancellation cache
  m_previousIntrinsicMatrix = cameraSystem()->cameraAtIndex(cameraIdx).intrinsicMatrix;
  m_previousDistCoeffs = cameraSystem()->cameraAtIndex(cameraIdx).distCoeffs;

  // Textures and RTs we use for captures
  m_fullGreyTex = rhi()->newTexture2D(cameraProvider()->streamWidth(), cameraProvider()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));
  m_fullGreyRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({m_fullGreyTex}));
  m_feedbackTex = rhi()->newTexture2D(cameraProvider()->streamWidth(), cameraProvider()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
  m_feedbackView.create(/*rows=*/ cameraProvider()->streamHeight(), /*columns=*/cameraProvider()->streamWidth(), CV_8UC4);

  m_feedbackRmsError = -1.0;
  m_incrementalUpdateInProgress = false;
}

CameraSystem::IntrinsicCalibrationContext::~IntrinsicCalibrationContext() {

}

CameraSystem::CalibrationContext::OverlayDistortionSpace CameraSystem::IntrinsicCalibrationContext::overlayDistortionSpace() const {
  if (inCalibrationPreviewMode())
    return kDistortionSpaceIntrinsic;

  return kDistortionSpaceUncorrected;
}

RHISurface::ptr CameraSystem::IntrinsicCalibrationContext::previewDistortionMapForCamera(size_t cameraIdx) const {
  if (inCalibrationPreviewMode())
    return cameraSystem()->cameraAtIndex(m_cameraIdx).intrinsicDistortionMap;

  return RHISurface::ptr();
}

void CameraSystem::IntrinsicCalibrationContext::renderStatusUI() {
  ImGui::Text("Camera %zu Intrinsic", m_cameraIdx);
  ImGui::Text("%zu samples", m_allCharucoCorners.size());
  if (m_feedbackRmsError < 0.0) {
    ImGui::Text("(insufficient/invalid data for preview)");
  } else if (m_incrementalUpdateInProgress) {
    ImGui::Text("Updating calibration...");
  } else {
    ImGui::Text("RMS error: %f", m_feedbackRmsError);
    ImGui::Text("FOV: %.2f x %.2f", m_feedbackFovX, m_feedbackFovY);
    ImGui::Text("Principal Point offset: %.2f, %.2f",
      static_cast<float>(cameraProvider()->streamWidth()  / 2) - m_feedbackPrincipalPoint.x,
      static_cast<float>(cameraProvider()->streamHeight() / 2) - m_feedbackPrincipalPoint.y);

    ImGui::Text("Per-View Errors");
    if (m_perViewErrors.rows > 0) {
      for (int i = 0; i < m_perViewErrors.rows; ++i) {
        ImGui::Text(" [%d] %f", i, m_perViewErrors.at<double>(i, 0));
      }
      if (ImGui::Button("Drop last sample")) {
        // drop sample
        m_allCharucoCorners.pop_back();
        m_allCharucoIds.pop_back();

        // update calibration feedback
        m_incrementalUpdateInProgress = true;
        FxThreading::runFunction([this]() { this->asyncUpdateIncrementalCalibration(); } );
      }
      if (ImGui::Button("Drop highest-error sample")) {
        double maxError = 0;
        int maxErrorRow = 0;
        for (int i = 0; i < m_perViewErrors.rows; ++i) {
          double e = m_perViewErrors.at<double>(i, 0);
          if (e > maxError) {
            e = maxError;
            maxErrorRow = i;
          }
        }

        // drop sample
        m_allCharucoCorners.erase(m_allCharucoCorners.begin() + maxErrorRow);
        m_allCharucoIds.erase(m_allCharucoIds.begin() + maxErrorRow);

        // update calibration feedback
        m_incrementalUpdateInProgress = true;
        FxThreading::runFunction([this]() { this->asyncUpdateIncrementalCalibration(); } );
      }
    }
  }
}


void CameraSystem::IntrinsicCalibrationContext::didAcceptCalibrationPreview() {
  // Calibration accepted. We don't need to do anything else here.
}

void CameraSystem::IntrinsicCalibrationContext::didRejectCalibrationPreview() {
  // Returning from preview to capture mode.
}

void CameraSystem::IntrinsicCalibrationContext::didCancelCalibrationSession() {
  // Restore previously saved calibration snapshot
  cameraSystem()->cameraAtIndex(m_cameraIdx).intrinsicMatrix = m_previousIntrinsicMatrix;
  cameraSystem()->cameraAtIndex(m_cameraIdx).distCoeffs = m_previousDistCoeffs;
  cameraSystem()->updateCameraIntrinsicDistortionParameters(m_cameraIdx);
}

bool CameraSystem::IntrinsicCalibrationContext::cookCalibrationDataForPreview() {
  if (m_feedbackRmsError < 0) {
    printf("CameraSystem::IntrinsicCalibrationContext::cookCalibrationDataForPreview(): calibration hasn't converged, can't generate a preview\n");
    return false; // incremental calibration data is invalid
  }

  // Recompute distortion parameters and update map
  cameraSystem()->updateCameraIntrinsicDistortionParameters(m_cameraIdx);
  return true;
}

void CameraSystem::IntrinsicCalibrationContext::processFrameCaptureMode() {
  cv::Mat viewFullRes = cameraSystem()->captureGreyscale(m_cameraIdx, m_fullGreyTex, m_fullGreyRT);

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

  if (m_incrementalUpdateInProgress)
    return; // don't allow capture while we're still updating

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
      for (size_t pixelIdx = 0; pixelIdx < (cameraProvider()->streamWidth() * cameraProvider()->streamHeight()); ++pixelIdx) {
        uint8_t* p = m_feedbackView.ptr(0) + (pixelIdx * 4);
        if (!(p[0] || p[1] || p[2])) {
          p[0] = p[1] = p[2] = viewFullRes.ptr(0)[pixelIdx];

        }
        p[3] = 0xff;
      }
      stbi_write_png(filename, cameraProvider()->streamWidth(), cameraProvider()->streamHeight(), 4, m_feedbackView.ptr(0), /*rowBytes=*/cameraProvider()->streamWidth() * 4);
      printf("Saved %s\n", filename);
    }

    m_incrementalUpdateInProgress = true;
    FxThreading::runFunction([this]() { this->asyncUpdateIncrementalCalibration(); } );
  }
}

void CameraSystem::IntrinsicCalibrationContext::asyncUpdateIncrementalCalibration() {
  try {
    Camera& c = cameraSystem()->cameraAtIndex(m_cameraIdx);

    cv::Mat stdDeviations;
    std::vector<float> reprojErrs;
    cv::Size imageSize(cameraProvider()->streamWidth(), cameraProvider()->streamHeight());
    int flags =
      cv::CALIB_FIX_PRINCIPAL_POINT |
      cv::CALIB_FIX_ASPECT_RATIO |
      cv::CALIB_RATIONAL_MODEL;

    c.intrinsicMatrix = cv::Mat::eye(3, 3, CV_64F);
    c.distCoeffs = cv::Mat::zeros(14, 1, CV_64F);


    m_feedbackRmsError = cv::aruco::calibrateCameraCharuco(m_allCharucoCorners, m_allCharucoIds,
                                   s_charucoBoard, imageSize,
                                   c.intrinsicMatrix, c.distCoeffs,
                                   cv::noArray(), cv::noArray(), stdDeviations, cv::noArray(),
                                   m_perViewErrors, flags);


    //printf("RMS error reported by calibrateCameraCharuco: %g\n", rms);
    //std::cout << "Camera " << m_cameraIdx << " Per-view error: " << std::endl << perViewErrors << std::endl;
    //std::cout << "Camera " << m_cameraIdx << " Matrix: " << std::endl << c.intrinsicMatrix << std::endl;
    //std::cout << "Camera " << m_cameraIdx << " Distortion coefficients: " << std::endl << c.distCoeffs << std::endl;

    double focalLength, aspectRatio; // not valid without a real aperture size, which we don't bother providing
    cv::calibrationMatrixValues(c.intrinsicMatrix, imageSize, 0.0, 0.0, m_feedbackFovX, m_feedbackFovY, focalLength, m_feedbackPrincipalPoint, aspectRatio);

  } catch (const std::exception& ex) {
    printf("Incremental calibration updated failed: %s\n", ex.what());
    m_feedbackRmsError = -1.0;
  }

  m_incrementalUpdateInProgress = false;
}

void CameraSystem::IntrinsicCalibrationContext::processFramePreviewMode() {
  // Look for charuco target and show reprojection errors.
  CameraSystem::Camera& c = cameraSystem()->cameraAtIndex(m_cameraIdx);

  cv::Mat viewFullRes = cameraSystem()->captureGreyscale(m_cameraIdx, m_fullGreyTex, m_fullGreyRT, c.intrinsicDistortionMap);

  std::vector<std::vector<cv::Point2f> > corners, rejected;
  std::vector<int> ids;

  std::vector<cv::Point2f> currentCharucoCorners;
  std::vector<int> currentCharucoIds;

  // Run ArUco marker detection
  // Note that we don't feed the camera distortion parameters to the aruco functions here, since the images we're operating on have already been undistorted.
  cv::aruco::detectMarkers(viewFullRes, s_charucoDictionary, corners, ids, cv::aruco::DetectorParameters::create(), rejected, c.optimizedMatrix, zeroDistortion);
  cv::aruco::refineDetectedMarkers(viewFullRes, s_charucoBoard, corners, ids, rejected, c.optimizedMatrix, zeroDistortion);

  // Find corners using detected markers
  if (!ids.empty()) {
    cv::aruco::interpolateCornersCharuco(corners, ids, viewFullRes, s_charucoBoard, currentCharucoCorners, currentCharucoIds, c.optimizedMatrix, zeroDistortion);
  }

  // Draw feedback view
  memset(m_feedbackView.ptr(0), 0, m_feedbackView.total() * 4);

  cv::Mat rvec, tvec;
  if (estimatePoseCharucoBoard(currentCharucoCorners, currentCharucoIds, s_charucoBoard, c.optimizedMatrix, zeroDistortion, rvec, tvec)) {
    // Enough markers are present to estimate the board pose

    std::vector<cv::Point2f> projPoints;
    cv::projectPoints(s_charucoBoard->chessboardCorners, rvec, tvec, c.optimizedMatrix, cv::noArray(), projPoints);

    for (size_t pointIdx = 0; pointIdx < currentCharucoIds.size(); ++pointIdx) {
      cv::Point2f corner = currentCharucoCorners[pointIdx];
      cv::Point2f rpCorner = projPoints[currentCharucoIds[pointIdx]];

      cv::Scalar color;
      float error = glm::length(glm::vec2(corner.x, corner.y) - glm::vec2(rpCorner.x, rpCorner.y));

      if (error < 0.25f) {
        color = cv::Scalar(0, 255, 0);
      }  else if (error < 0.75f) {
        color = cv::Scalar(0, 0, 255);
      } else {
        color = cv::Scalar(255, 0, 0);
      }


      // draw reference corner mark
      cv::rectangle(m_feedbackView, corner - cv::Point2f(5, 5), corner + cv::Point2f(5, 5), color, 1, cv::LINE_AA);

      // draw reprojected corner mark
      cv::rectangle(m_feedbackView, rpCorner - cv::Point2f(3, 3), rpCorner + cv::Point2f(3, 3), color, 1, cv::LINE_AA);

      // draw error callout
      char textbuf[16];
      sprintf(textbuf, "%.2fpx", error);
      cv::putText(m_feedbackView, textbuf, corner + cv::Point2f(7, -7), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    } // point loop
  }

  rhi()->loadTextureData(m_feedbackTex, kVertexElementTypeUByte4N, m_feedbackView.ptr(0));
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

  CameraSystem::View& v = cameraSystem()->viewAtIndex(m_viewIdx);
  m_calibState = new CharucoMultiViewCalibration(cameraSystem(), {v.cameraIndices[0], v.cameraIndices[1]});
  m_calibState->m_undistortCapturedViews = false; // operate in native space

  m_feedbackTx = glm::vec3(0.0f);
  m_feedbackRx = glm::vec3(0.0f);
  m_feedbackRmsError = 0.0;
}

CameraSystem::StereoCalibrationContext::~StereoCalibrationContext() {
  delete m_calibState;
}

void CameraSystem::StereoCalibrationContext::renderStatusUI() {
  ImGui::Text("View %zu Stereo", m_viewIdx);
  ImGui::Text("%zu samples", m_calibState->m_objectPoints.size());
  ImGui::Text("Est. Tx  (mm): %.3g %.3g %.3g", m_feedbackTx[0] * 1000.0f, m_feedbackTx[1] * 1000.0f, m_feedbackTx[2] * 1000.0f);
  ImGui::Text("Est. Rx (deg): %.3g %.3g %.3g", glm::degrees(m_feedbackRx[0]), glm::degrees(m_feedbackRx[1]), glm::degrees(m_feedbackRx[2]));
  ImGui::Text("RMS Error: %f", m_feedbackRmsError);
  ImGui::Text("Stereo ROI (L): [%u x %u from (%u, %u)", m_feedbackValidROI[0].width, m_feedbackValidROI[0].height, m_feedbackValidROI[0].x, m_feedbackValidROI[0].y);
  ImGui::Text("Stereo ROI (R): [%u x %u from (%u, %u)", m_feedbackValidROI[1].width, m_feedbackValidROI[1].height, m_feedbackValidROI[1].x, m_feedbackValidROI[1].y);
  ImGui::Text("Per-View Errors");
  if (m_perViewErrors.rows > 0) {
    for (int i = 0; i < m_perViewErrors.rows; ++i) {
      ImGui::Text(" [%d] %f, %f", i, m_perViewErrors.at<double>(i, 0), m_perViewErrors.at<double>(i, 1));
    }
    if (ImGui::Button("Drop highest-error sample")) {
      double maxError = 0;
      int maxErrorRow = 0;
      for (int i = 0; i < m_perViewErrors.rows; ++i) {
        double e = std::max<double>(m_perViewErrors.at<double>(i, 0), m_perViewErrors.at<double>(i, 1));
        if (e > maxError) {
          e = maxError;
          maxErrorRow = i;
        }
      }

      m_calibState->dropSampleAtIndex(maxErrorRow);
      internalUpdateCaptureState();
    }
  }
}


void CameraSystem::StereoCalibrationContext::processFrameCaptureMode() {
  if (m_calibState->processFrame(captureRequested())) {
    // capture request succeeded if return is true
    acknowledgeCaptureRequest();

    internalUpdateCaptureState();

#if 0 // TODO fix this
    if (shouldSaveCalibrationImages()) {
      char filename[128];
      sprintf(filename, "calib_view_%zu_%02u_overlay.png", m_viewIdx, m_objectPoints.size());

      // composite with the greyscale view and fix the alpha channel before writing
      for (size_t pixelIdx = 0; pixelIdx < (cameraProvider()->streamWidth() * 2) * cameraProvider()->streamHeight(); ++pixelIdx) {
        uint8_t* p = feedbackViewStereo.ptr(0) + (pixelIdx * 4);
        if (!(p[0] || p[1] || p[2])) {
          p[0] = p[1] = p[2] = viewFullResStereo.ptr(0)[pixelIdx];

        }
        p[3] = 0xff;
      }
      stbi_write_png(filename, cameraProvider()->streamWidth()*2, cameraProvider()->streamHeight(), 4, feedbackViewStereo.ptr(0), /*rowBytes=*/cameraProvider()->streamWidth()*2 * 4);
      printf("Saved %s\n", filename);
    }
#endif
  }
}

void CameraSystem::StereoCalibrationContext::internalUpdateCaptureState() {
  cv::Mat feedbackE, feedbackF;
  cv::Mat feedbackTx, feedbackRx;

  View& v = cameraSystem()->viewAtIndex(m_viewIdx);
  Camera& leftC = cameraSystem()->cameraAtIndex(v.cameraIndices[0]);
  Camera& rightC = cameraSystem()->cameraAtIndex(v.cameraIndices[1]);

  int flags =
    cv::CALIB_USE_INTRINSIC_GUESS | // load previously-established intrinsics
    cv::CALIB_FIX_INTRINSIC |       // and don't modify them
    cv::CALIB_FIX_PRINCIPAL_POINT |
    cv::CALIB_FIX_ASPECT_RATIO |
    cv::CALIB_RATIONAL_MODEL;

  m_feedbackRmsError = cv::stereoCalibrate(m_calibState->m_objectPoints,
    m_calibState->m_calibrationPoints[0], m_calibState->m_calibrationPoints[1],
    leftC.intrinsicMatrix, leftC.distCoeffs,
    rightC.intrinsicMatrix, rightC.distCoeffs,
    cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()),
    feedbackRx, feedbackTx, feedbackE, feedbackF, m_perViewErrors, flags);

  m_feedbackTx = glmVec3FromCV(feedbackTx);
  glm::mat4 rx = glmMat3FromCVMatrix(feedbackRx);
  glm::extractEulerAngleXYZ(rx, m_feedbackRx[0], m_feedbackRx[1], m_feedbackRx[2]);

  cv::Mat feedbackRect[2], feedbackProj[2], feedbackQ;
  cv::Rect stereoValidROI[2];

  cv::stereoRectify(
    leftC.intrinsicMatrix, leftC.distCoeffs,
    rightC.intrinsicMatrix, rightC.distCoeffs,
    cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()),
    feedbackRx, feedbackTx,
    feedbackRect[0], feedbackRect[1],
    feedbackProj[0], feedbackProj[1],
    feedbackQ,
    /*flags=*/cv::CALIB_ZERO_DISPARITY, /*alpha=*/ -1.0f, cv::Size(),
    &m_feedbackValidROI[0], &m_feedbackValidROI[1]);

}


void CameraSystem::StereoCalibrationContext::processFramePreviewMode() {

}

bool CameraSystem::StereoCalibrationContext::cookCalibrationDataForPreview() {
  try {
    // Samples collected, run calibration
    cv::Mat E, F;
    cv::Mat perViewErrors;

    View& v = cameraSystem()->viewAtIndex(m_viewIdx);
    Camera& leftC = cameraSystem()->cameraAtIndex(v.cameraIndices[0]);
    Camera& rightC = cameraSystem()->cameraAtIndex(v.cameraIndices[1]);

    int flags =
      cv::CALIB_USE_INTRINSIC_GUESS | // load previously-established intrinsics
      cv::CALIB_FIX_INTRINSIC |       // and don't modify them
      cv::CALIB_FIX_PRINCIPAL_POINT |
      cv::CALIB_FIX_ASPECT_RATIO |
      cv::CALIB_RATIONAL_MODEL;

    double rms = cv::stereoCalibrate(m_calibState->m_objectPoints,
      m_calibState->m_calibrationPoints[0], m_calibState->m_calibrationPoints[1],
      leftC.intrinsicMatrix, leftC.distCoeffs,
      rightC.intrinsicMatrix, rightC.distCoeffs,
      cv::Size(cameraProvider()->streamWidth(), cameraProvider()->streamHeight()),
      v.stereoRotation, v.stereoTranslation, E, F, perViewErrors, flags);

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
  // Returning from preview to capture mode.
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
  return m_calibState->m_feedbackTex[index];
}





CameraSystem::StereoViewOffsetCalibrationContext::StereoViewOffsetCalibrationContext(CameraSystem* cs, size_t referenceViewIdx, size_t viewIdx) : CameraSystem::CalibrationContextStateMachineBase(cs), m_referenceViewIdx(referenceViewIdx), m_viewIdx(viewIdx) {
  m_useLinearRemap = false;

  // Store cancellation cache
  m_previousViewTranslation = cameraSystem()->viewAtIndex(viewIdx).viewTranslation;
  m_previousViewRotation = cameraSystem()->viewAtIndex(viewIdx).viewRotation;

  CameraSystem::View& rv = cameraSystem()->viewAtIndex(m_referenceViewIdx);
  CameraSystem::View& v = cameraSystem()->viewAtIndex(m_viewIdx);

  m_refCalibState = new CharucoMultiViewCalibration(cameraSystem(),
    {rv.cameraIndices[0], rv.cameraIndices[1]},
    {referenceViewIdx, referenceViewIdx});

  m_tgtCalibState = new CharucoMultiViewCalibration(cameraSystem(),
    {v.cameraIndices[0], v.cameraIndices[1]},
    {viewIdx, viewIdx});

  m_tgt2ref = glm::mat4(1.0f);
}

CameraSystem::StereoViewOffsetCalibrationContext::~StereoViewOffsetCalibrationContext() {
  delete m_refCalibState;
  delete m_tgtCalibState;
}

void CameraSystem::StereoViewOffsetCalibrationContext::renderStatusUI() {
  ImGui::Text("View %zu Stereo Offset", m_viewIdx);
  ImGui::Text("%zu points", m_refPoints.size());

  ImGui::Text("Tx  (mm): %.1f %.1f %.1f", m_tgt2ref[3][0] * 1000.0f, m_tgt2ref[3][1] * 1000.0f, m_tgt2ref[3][2] * 1000.0f);
  ImGui::Text("RMS Error (mm): %.2f", m_rmsError * 1000.0f);

  float rx, ry, rz;
  glm::extractEulerAngleXYZ(m_tgt2ref, rx, ry, rz);
  ImGui::Text("Rx (deg): %.2f %.2f %.2f", glm::degrees(rx), glm::degrees(ry), glm::degrees(rz));

  ImGui::Checkbox("Use linear remap", &m_useLinearRemap);
  ImGui::Text("(Less accurate, but does not require common board-points across all views)");

}

template <typename T> static std::vector<T> flattenVector(const std::vector<std::vector<T> >& in) {
  std::vector<T> res;
  size_t s = 0;
  for (size_t i = 0; i < in.size(); ++i) {
    s += in[i].size();
  }
  res.reserve(s);
  for (size_t i = 0; i < in.size(); ++i) {
    for (size_t j = 0; j < in[i].size(); ++j) {
      res.push_back(in[i][j]);
    }
  }
  return res;
}

int triangulationDisparityScaleInv = 16;

std::vector<glm::vec3> getTriangulatedPointsForView(CameraSystem* cameraSystem, size_t viewIdx, const std::vector<std::vector<cv::Point2f> >& leftCalibrationPoints, const std::vector<std::vector<cv::Point2f> >& rightCalibrationPoints) {
  std::vector<glm::vec3> res;

  auto lp = flattenVector(leftCalibrationPoints);
  auto rp = flattenVector(rightCalibrationPoints);
  assert(lp.size() == rp.size());

  if (lp.empty())
    return res;

  res.resize(lp.size());

  // Input points (m_calibState->m_calibrationPoints) have been captured in distortion-corrected, stereo-remapped space

#if 1
  CameraSystem::View& view = cameraSystem->viewAtIndex(viewIdx);
  glm::mat3 R1inv = glm::inverse(glmMat3FromCVMatrix(view.stereoRectification[0]));
  glm::mat4 Q = glmMat4FromCVMatrix(view.stereoDisparityToDepth);
  float CameraDistanceMeters = glm::length(glm::vec3(view.stereoTranslation.at<double>(0), view.stereoTranslation.at<double>(1), view.stereoTranslation.at<double>(2)));

  float dispScale = 16.0f / static_cast<float>(triangulationDisparityScaleInv);

  bool isVerticalStereo = view.isVerticalStereo();

  for (size_t pointIdx = 0; pointIdx < lp.size(); ++pointIdx) {
    float x = lp[pointIdx].x;
    float y = lp[pointIdx].y;

    float fDisp;
    if (isVerticalStereo)
      fDisp = fabs(rp[pointIdx].y - lp[pointIdx].y) / dispScale;
    else
      fDisp = fabs(rp[pointIdx].x - lp[pointIdx].x) / dispScale;

    float lz = (Q[2][3] * CameraDistanceMeters) / fDisp;
    float ly = (y + Q[1][3]) / Q[2][3];
    float lx = (x + Q[0][3]) / Q[2][3];
    lx *= lz;
    ly *= lz;

    res[pointIdx] = R1inv * glm::vec3(lx, -ly, -lz);
  }


#else

  // the coordsys for this function needs to have X and Y negated to match

  cv::Mat triangulatedPointsH;
  cv::triangulatePoints(view.stereoProjection[0], view.stereoProjection[1], lp, rp, triangulatedPointsH);

  // output matrix shape is {channels=1 rows=4 cols=#points}
  // convertPointsFromHomogeneous wants {channels=4 rows=#points cols=1}

  cv::Mat inH = triangulatedPointsH.t(); // transpose to make point components sequential in each row
  inH.reshape(/*channels=*/ 4); // reshape to convert columns to components
  //triangulatedPointsH.reshape(/*channels=*/4, /*rows=*/1);

  cv::Mat dest;
  cv::convertPointsFromHomogeneous(inH, dest);
#endif

  return res;
}

std::vector<glm::vec3> transformBoardPointsForView(const glm::mat4& transform) {
  std::vector<glm::vec3> res;
  res.resize(s_charucoBoard->chessboardCorners.size());

  for (size_t pointIdx = 0; pointIdx < res.size(); ++pointIdx) {
    res[pointIdx] = glm::vec3(transform * glm::vec4(glmVec3FromCV(s_charucoBoard->chessboardCorners[pointIdx]), 1.0f));
  }
  return res;
}

cv::Vec3f centroid(const cv::Mat& m) {
  assert((m.cols == 1 && m.type() == CV_32FC3) || (m.cols == 3 && m.type() == CV_32FC1));

  cv::Vec3f res = cv::Vec3f::all(0);

  for (int i = 0; i < m.rows; ++i) {
    res += m.at<cv::Vec3f>(i);
  }

  return res.mul(cv::Vec3f::all(1.0f / static_cast<float>(m.rows)));
}

glm::vec3 centroid(const std::vector<glm::vec3>& points) {

  glm::vec3 res = glm::vec3(0.0f);

  for (size_t i = 0; i < points.size(); ++i) {
    res += points[i];
  }

  return res * glm::vec3(1.0f / static_cast<float>(points.size()));
}

float computePointSetLinearTransform(const std::vector<glm::vec3>& pVec1, const std::vector<glm::vec3>& pVec2, glm::mat4& outTransform) {
  // Algorithm refs:
  // http://nghiaho.com/?page_id=671
  // https://github.com/nghiaho12/rigid_transform_3D

  assert(pVec1.size() == pVec2.size());

  glm::vec3 c1 = centroid(pVec1);
  glm::vec3 c2 = centroid(pVec2);

  // Matrix multiply requires 1-channel 3-column matrix
  cv::Mat p1Centered(pVec1.size(), 3, CV_32F);
  cv::Mat p2Centered(pVec1.size(), 3, CV_32F);

  for (int row = 0; row < pVec1.size(); ++row) {
    // glm::vec3 and 3xCV_32F have identical in-memory layout, so we can treat each row as a glm::vec3 for convenience
    p1Centered.at<glm::vec3>(row) = pVec1[row] - c1;
    p2Centered.at<glm::vec3>(row) = pVec2[row] - c2;
  }

  // covariance matrix
  cv::Mat H = p1Centered.t() * p2Centered;
  assert(H.rows == 3 && H.cols == 3);

  cv::Mat w, u, vt;
  cv::SVD::compute(H, w, u, vt);

  cv::Mat R = vt.t() * u.t();

  if (cv::determinant(R) < 0) {
    //printf("det < 0\n");
    // correct for reflection case
    for (int i = 0; i < 3; ++i) {
      vt.at<float>(2, i) *= -1.0f;
    }
    R = vt.t() * u.t();
    assert(cv::determinant(R) >= 0.0);
  }

  outTransform = glm::translate(c1) * glmMat4FromCVMatrix(R) * glm::translate(-c2);

  // compute error: transform each of p2 by the computed transform and compare to the corresponding p1 point
  float errorAccum = 0.0f;

  for (size_t pIdx = 0; pIdx < pVec1.size(); ++pIdx) {
    glm::vec3 p2t = glm::vec3(outTransform * glm::vec4(pVec2[pIdx], 1.0f));
    float err = glm::length(pVec1[pIdx] - p2t);
    errorAccum += sqrtf(err);
  }
  errorAccum /= static_cast<float>(pVec1.size());
  errorAccum *= errorAccum;
  return errorAccum;
}

std::vector<glm::vec3> recoverViewSpaceBoardPoints(CameraSystem* cameraSystem, size_t viewIdx, CharucoMultiViewCalibration* calibState) {
  std::vector<glm::vec3> res;

  std::vector<glm::vec3> viewP = getTriangulatedPointsForView(cameraSystem, viewIdx, calibState->m_calibrationPoints[0], calibState->m_calibrationPoints[1]);
  if (viewP.empty())
    return res; // no points

  std::vector<int> ids = flattenVector(calibState->m_objectIds);

  std::vector<glm::vec3> objectP;
  for (int id : ids) {
    objectP.push_back(glmVec3FromCV(s_charucoBoard->chessboardCorners[id]));
  }

  glm::mat4 linearRemapXf;
  float linearRemapError = computePointSetLinearTransform(viewP, objectP, linearRemapXf);
  if (linearRemapError < 0) {
    return res; // computation failed
  }

  // use computed transform to map all board points into view space
  for (const cv::Point3f& p : s_charucoBoard->chessboardCorners) {
    res.push_back(glm::vec3(linearRemapXf * glm::vec4(glmVec3FromCV(p), 1.0f)));
  }

  return res;
}

void CameraSystem::StereoViewOffsetCalibrationContext::processFrameCaptureMode() {

  // we don't need the calibration states to save point vectors, we just want the board orientations
  m_refCalibState->reset();
  m_tgtCalibState->reset();

  bool didCapRef = m_refCalibState->processFrame(captureRequested());
  bool didCapTgt = m_tgtCalibState->processFrame(captureRequested());

  if (!(didCapRef && didCapTgt))
    return;

  // capture request succeeded if return is true
  acknowledgeCaptureRequest();


  if (m_useLinearRemap) {
    // Recover the board pose using visible points from each view-pair, then use the board pose to estimate the position of all points.
    // Requires the board to be visible in all views, but does not require the same points to be visible.

    std::vector<glm::vec3> refVP = recoverViewSpaceBoardPoints(cameraSystem(), m_referenceViewIdx, m_refCalibState);
    std::vector<glm::vec3> tgtVP = recoverViewSpaceBoardPoints(cameraSystem(), m_viewIdx, m_tgtCalibState);

    if (refVP.empty() || tgtVP.empty())
      return; // mapping failed

    // Keep only the points that exist in at least one view -- that might help reduce extrapolation error?
    std::set<int> idSet;

    for (int id : flattenVector(m_refCalibState->m_objectIds))
      idSet.insert(id);
    for (int id : flattenVector(m_tgtCalibState->m_objectIds))
      idSet.insert(id);

    for (int id : idSet) {
      m_refPoints.push_back(refVP[id]);
      m_tgtPoints.push_back(tgtVP[id]);
    }
  } else {
    // Use direct matching of points between views. Requires common points to be visible in all views.

    std::vector<glm::vec3> tgtViewP = getTriangulatedPointsForView(cameraSystem(), m_viewIdx, m_tgtCalibState->m_calibrationPoints[0], m_tgtCalibState->m_calibrationPoints[1]);
    std::vector<int> tgtIds = flattenVector(m_tgtCalibState->m_objectIds);

    std::vector<glm::vec3> refViewP = getTriangulatedPointsForView(cameraSystem(), m_referenceViewIdx, m_refCalibState->m_calibrationPoints[0], m_refCalibState->m_calibrationPoints[1]);
    std::vector<int> refIds = flattenVector(m_refCalibState->m_objectIds);

    for (size_t i = 0; i < refIds.size(); ++i) {
      for (size_t j = 0; j < tgtIds.size(); ++j) {
        if (refIds[i] == tgtIds[j]) {
          m_refPoints.push_back(refViewP[i]);
          m_tgtPoints.push_back(tgtViewP[j]);
          break;
        }
      }
    }
  }

#if 0
  // formatted for Mathematica
  printf("StereoViewOffsetCalibrationContext::processFrameCaptureMode(): %d points:\n", refPoints.rows);

  printf("{ \n");
  for (size_t i = 0; i < refPoints.rows; ++i) {
    float* rp = refPoints.ptr<float>(i);
    printf("  {%f, %f, %f}, \n",
      rp[0], rp[1], rp[2]);
  }
  printf("}, {\n");
  for (size_t i = 0; i < refPoints.rows; ++i) {
    float* tp = tgtPoints.ptr<float>(i);
    printf("  {%f, %f, %f}, \n",
      tp[0], tp[1], tp[2]);
  }
  printf("}\n");
#endif


  m_rmsError = computePointSetLinearTransform(m_refPoints, m_tgtPoints, m_tgt2ref);
}

void CameraSystem::StereoViewOffsetCalibrationContext::processFramePreviewMode() {

}

bool CameraSystem::StereoViewOffsetCalibrationContext::cookCalibrationDataForPreview() {
  // We've been updating the calibration data incrementally every frame -- nothing to do here.
  return true;
}

void CameraSystem::StereoViewOffsetCalibrationContext::didAcceptCalibrationPreview() {
  // Calibration accepted -- store the view translation and rotation.
  View& v = cameraSystem()->viewAtIndex(m_viewIdx);
  v.viewTranslation = glm::vec3(m_tgt2ref[3]);

  glm::vec3 rv;
  glm::extractEulerAngleXYZ(m_tgt2ref, rv[0], rv[1], rv[2]);
  v.viewRotation = glm::degrees(rv);
}

void CameraSystem::StereoViewOffsetCalibrationContext::didRejectCalibrationPreview() {
  // Returning from preview to capture mode.
}

void CameraSystem::StereoViewOffsetCalibrationContext::didCancelCalibrationSession() {
  // Restore previously saved calibration snapshot
  cameraSystem()->viewAtIndex(m_viewIdx).viewTranslation = m_previousViewTranslation;
  cameraSystem()->viewAtIndex(m_viewIdx).viewRotation = m_previousViewRotation;
}

bool CameraSystem::StereoViewOffsetCalibrationContext::requiresStereoRendering() const {
  return true;
}

RHISurface::ptr CameraSystem::StereoViewOffsetCalibrationContext::overlaySurfaceAtIndex(size_t index) {
  assert(index < 4);
  if (index < 2)
    return m_refCalibState->m_feedbackTex[index];
  else
    return m_tgtCalibState->m_feedbackTex[index - 2];
}
