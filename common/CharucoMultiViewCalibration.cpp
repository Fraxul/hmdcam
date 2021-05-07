#include "common/CharucoMultiViewCalibration.h"
#include "common/CameraSystem.h"
#include "common/FxThreading.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/calib3d.hpp>
#include <set>

extern cv::Ptr<cv::aruco::Dictionary> s_charucoDictionary; // in CameraSystem
extern cv::Ptr<cv::aruco::CharucoBoard> s_charucoBoard; // in CameraSystem
static const cv::Mat zeroDistortion = cv::Mat::zeros(1, 5, CV_32FC1);

CharucoMultiViewCalibration::CharucoMultiViewCalibration(CameraSystem* cs_, const std::vector<size_t>& cameraIds_, const std::vector<size_t>& cameraStereoViewIds_) : m_undistortCapturedViews(true), m_enableFeedbackView(true), m_cameraSystem(cs_), m_cameraIds(cameraIds_) {

  m_fullGreyTex.resize(cameraCount());
  m_fullGreyRT.resize(cameraCount());
  m_feedbackTex.resize(cameraCount());
  m_feedbackView.resize(cameraCount());

  m_calibrationPoints.resize(cameraCount());

  m_cameraStereoViewIds.resize(m_cameraIds.size(), -1);
  for (size_t cameraIdx = 0; cameraIdx < cameraStereoViewIds_.size(); ++cameraIdx) {
    ssize_t x = cameraStereoViewIds_[cameraIdx];
    if (x >= 0 && x < cameraSystem()->views()) {
      CameraSystem::View& v = cameraSystem()->viewAtIndex(x);
      assert(v.isStereo && (v.cameraIndices[0] == m_cameraIds[cameraIdx] || v.cameraIndices[1] == m_cameraIds[cameraIdx]));
      m_cameraStereoViewIds[cameraIdx] = x;
    }
  }

  for (size_t cameraIdx = 0; cameraIdx < cameraCount(); ++cameraIdx) {
    m_fullGreyTex[cameraIdx] = rhi()->newTexture2D(cameraProvider()->streamWidth(), cameraProvider()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));
    m_fullGreyRT[cameraIdx] = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({m_fullGreyTex[cameraIdx]}));
    m_feedbackTex[cameraIdx] = rhi()->newTexture2D(cameraProvider()->streamWidth(), cameraProvider()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    m_feedbackView[cameraIdx].create(/*rows=*/ cameraProvider()->streamHeight(), /*columns=*/cameraProvider()->streamWidth(), CV_8UC4);
  }

}

ICameraProvider* CharucoMultiViewCalibration::cameraProvider() const {
  return cameraSystem()->cameraProvider();
}

bool CharucoMultiViewCalibration::processFrame(bool captureRequested) {
  // Capture and undistort camera views.
  std::vector<cv::Mat> eyeFullRes(cameraCount());
  for (size_t cameraIdx = 0; cameraIdx < cameraCount(); ++cameraIdx) {

    RHISurface::ptr distortionMap;

    if (m_undistortCapturedViews) {
      if (m_cameraStereoViewIds[cameraIdx] >= 0) {
        // Stereo distortion correction
        CameraSystem::View& v = cameraSystem()->viewAtIndex(m_cameraStereoViewIds[cameraIdx]);
        distortionMap = (m_cameraIds[cameraIdx] == v.cameraIndices[0]) ? v.stereoDistortionMap[0] : v.stereoDistortionMap[1];
        assert(distortionMap);
      } else {
        // Intrinsic distortion correction
        distortionMap = cameraSystem()->cameraAtIndex(m_cameraIds[cameraIdx]).intrinsicDistortionMap;
      }
    }

    eyeFullRes[cameraIdx] = cameraSystem()->captureGreyscale(m_cameraIds[cameraIdx], m_fullGreyTex[cameraIdx], m_fullGreyRT[cameraIdx], distortionMap);
  }

  std::vector<std::vector<std::vector<cv::Point2f> > > corners(cameraCount());
  std::vector<std::vector<std::vector<cv::Point2f> > > rejected(cameraCount());
  std::vector<std::vector<int> > ids(cameraCount());

  std::vector<std::vector<cv::Point2f> > currentCharucoCornerPoints(cameraCount());
  std::vector<std::vector<int> > currentCharucoCornerIds(cameraCount());

  // Run ArUco marker detection
  // Note that we don't feed the camera distortion parameters to the aruco functions here, since the images we're operating on have already been undistorted.
  FxThreading::runArrayTask(0, cameraCount(), [&](size_t cameraIdx) {
    CameraSystem::Camera& c = cameraSystem()->cameraAtIndex(m_cameraIds[cameraIdx]);

    cv::aruco::detectMarkers(eyeFullRes[cameraIdx], s_charucoDictionary, corners[cameraIdx], ids[cameraIdx], cv::aruco::DetectorParameters::create(), rejected[cameraIdx], c.optimizedMatrix, zeroDistortion);
    cv::aruco::refineDetectedMarkers(eyeFullRes[cameraIdx], s_charucoBoard, corners[cameraIdx], ids[cameraIdx], rejected[cameraIdx], c.optimizedMatrix, zeroDistortion);

    // Find chessboard corners using detected markers
    if (!ids[cameraIdx].empty()) {
      cv::aruco::interpolateCornersCharuco(corners[cameraIdx], ids[cameraIdx], eyeFullRes[cameraIdx], s_charucoBoard, currentCharucoCornerPoints[cameraIdx], currentCharucoCornerIds[cameraIdx], c.optimizedMatrix, zeroDistortion);
    }
  });

  // Find set of chessboard corners present in all views
  std::set<int> commonCornerIds;
  {
    // Start with all ids in the first view
    for (size_t i = 0; i < currentCharucoCornerIds[0].size(); ++i) {
      commonCornerIds.insert(currentCharucoCornerIds[0][i]);
    }
    // And subtract ids that are missing in subsequent views
    for (size_t cameraIdx = 1; cameraIdx < cameraCount(); ++cameraIdx) {
      // Populate set for current camera
      std::set<int> currentCameraSet;
      for (size_t i = 0; i < currentCharucoCornerIds[cameraIdx].size(); ++i) {
        currentCameraSet.insert(currentCharucoCornerIds[cameraIdx][i]);
      }

      // Remove elements from common set not found in current camera set
      for (std::set<int>::iterator it = commonCornerIds.begin(); it != commonCornerIds.end(); ) {
        if (currentCameraSet.find(*it) == currentCameraSet.end()) {
          it = commonCornerIds.erase(it);
        } else {
          ++it;
        }
      }
    }
  }

  // Require at least 6 corners visibile to both cameras to consider this frame
  bool foundOverlap = commonCornerIds.size() >= 6;

  // Filter the eye corner sets to only commonly visible corners, which we will later feed to stereoCalibrate
  std::vector<cv::Point3f> thisFrameBoardRefCorners;
  std::vector<std::vector<cv::Point2f> > thisFrameImageCorners(cameraCount());

  for (std::set<int>::const_iterator corner_it = commonCornerIds.begin(); corner_it != commonCornerIds.end(); ++corner_it) {
    int cornerId = *corner_it;

    for (size_t cameraIdx = 0; cameraIdx < cameraCount(); ++cameraIdx) {
      for (size_t eyeCornerIdx = 0; eyeCornerIdx < currentCharucoCornerIds[cameraIdx].size(); ++eyeCornerIdx) {
        if (currentCharucoCornerIds[cameraIdx][eyeCornerIdx] == cornerId) {
          thisFrameImageCorners[cameraIdx].push_back(currentCharucoCornerPoints[cameraIdx][eyeCornerIdx]);
          break;
        }
      }
    }

    // save the corner point in board space from the board definition
    thisFrameBoardRefCorners.push_back(s_charucoBoard->chessboardCorners[cornerId]);
  }

  for (size_t cameraIdx = 0; cameraIdx < cameraCount(); ++cameraIdx) {
    assert(thisFrameBoardRefCorners.size() == thisFrameImageCorners[cameraIdx].size());
  }


  // Draw feedback views
  if (m_enableFeedbackView) {
    FxThreading::runArrayTask(0, cameraCount(), [&](size_t cameraIdx) {
      memset(m_feedbackView[cameraIdx].ptr(0), 0, m_feedbackView[cameraIdx].total() * 4);

      if (!corners[cameraIdx].empty()) {
        cv::aruco::drawDetectedMarkers(m_feedbackView[cameraIdx], corners[cameraIdx]);
      }

      // Borrowed from cv::aruco::drawDetectedCornersCharuco -- modified to switch the color per-marker to indicate stereo visibility
      for(size_t cornerIdx = 0; cornerIdx < currentCharucoCornerIds[cameraIdx].size(); ++cornerIdx) {
        cv::Point2f corner = currentCharucoCornerPoints[cameraIdx][cornerIdx];
        int id = currentCharucoCornerIds[cameraIdx][cornerIdx];

        // grey for mono points
        cv::Scalar cornerColor = cv::Scalar(127, 127, 127);
        if (commonCornerIds.find(id) != commonCornerIds.end()) {
          // red for stereo points
          cornerColor = cv::Scalar(255, 0, 0);
        }

        // draw first corner mark
        cv::rectangle(m_feedbackView[cameraIdx], corner - cv::Point2f(3, 3), corner + cv::Point2f(3, 3), cornerColor, 1, cv::LINE_AA);

        // draw ID
        char idbuf[16];
        sprintf(idbuf, "id=%u", id);
        cv::putText(m_feedbackView[cameraIdx], idbuf, corner + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cornerColor, 2);
      }
    });

    for (size_t cameraIdx = 0; cameraIdx < cameraCount(); ++cameraIdx) {
      rhi()->loadTextureData(m_feedbackTex[cameraIdx], kVertexElementTypeUByte4N, m_feedbackView[cameraIdx].ptr(0));
    }
  }


  // Handle capture requests
  if (foundOverlap && captureRequested) {
    m_objectPoints.push_back(thisFrameBoardRefCorners);
    for (size_t cameraIdx = 0; cameraIdx < cameraCount(); ++cameraIdx) {
      m_calibrationPoints[cameraIdx].push_back(thisFrameImageCorners[cameraIdx]);
    }
    return true;
  }

  return false;
}

void CharucoMultiViewCalibration::reset() {
  m_objectPoints.clear();
  for (size_t cameraIdx = 0; cameraIdx < cameraCount(); ++cameraIdx) {
    m_calibrationPoints[cameraIdx].clear();
  }
}

