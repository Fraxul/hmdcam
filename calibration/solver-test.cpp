#include <cstdio>
#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>

#include "common/Timing.h"

#include <industrial_calibration/optimizations/camera_intrinsic.h>
#include <ceres/ceres.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/aruco_calib.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>

#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/transform.hpp>

#include "common/FxThreading.cpp" // just inlining FxThreading in here for build simplicity

static const cv::Mat zeroDistortion = cv::Mat::zeros(1, 14, CV_64F);

// ChAruCo target pattern config
const cv::aruco::PredefinedDictionaryType s_charucoDictionaryName = cv::aruco::DICT_5X5_100;
const unsigned int s_charucoBoardSquareCountX = 12;
const unsigned int s_charucoBoardSquareCountY = 9;
const float s_charucoBoardSquareSideLengthMeters = 0.060f;
// markers are 7x7 pixels, squares are 9x9 pixels (add 1px border), so the marker size is 7/9 of the square size
const float s_charucoBoardMarkerSideLengthMeters = s_charucoBoardSquareSideLengthMeters * (7.0f / 9.0f);

cv::aruco::Dictionary charucoDictionary() { return cv::aruco::getPredefinedDictionary(s_charucoDictionaryName); }
cv::Ptr<cv::aruco::CharucoBoard> s_charucoBoard;
cv::aruco::DetectorParameters s_detectorParams;

cv::aruco::CharucoDetector createCharucoDetector(cv::Mat cameraMatrix = cv::Mat(), cv::Mat distCoeffs = cv::Mat()) {
  cv::aruco::CharucoParameters chParams;
  chParams.cameraMatrix = cameraMatrix;
  chParams.distCoeffs = distCoeffs;
  chParams.tryRefineMarkers = true;
  return cv::aruco::CharucoDetector(*s_charucoBoard, chParams, s_detectorParams);
}


// Represents observed point correspondances between image and target/object space in a single captured image
struct Observation {
  cv::Mat image;

  size_t pointCount() const { return objectPoints.size(); }
  bool empty() const { return objectPoints.empty(); }

  industrial_calibration::Correspondence2D3D::Set correspondenceSet() {
    industrial_calibration::Correspondence2D3D::Set corrSet;
    corrSet.resize(objectPoints.size());

    for (size_t pointIdx = 0; pointIdx < objectPoints.size(); ++pointIdx) {
      industrial_calibration::Correspondence2D3D& corr = corrSet[pointIdx];
      corr.in_image = Eigen::Vector2d(imagePoints[pointIdx].x, imagePoints[pointIdx].y);
      corr.in_target = Eigen::Vector3d(objectPoints[pointIdx].x, objectPoints[pointIdx].y, objectPoints[pointIdx].z);
    }
    return corrSet;
  }

  std::vector<cv::Point3f> objectPoints; // Points in object space, on the calibration target
  std::vector<cv::Point2f> imagePoints; // Points in image space
};

struct ViewCalibrationData {
  cv::Size imageSize;
  std::vector<Observation> observations;

};


struct MultiViewCalibrationData {
  std::vector<ViewCalibrationData> views;
};


int main(int argc, char** argv) {
  FxThreading::detail::init();

  // Initialize ChAruCo data on first use
  if (!s_charucoBoard)
    s_charucoBoard = new cv::aruco::CharucoBoard(cv::Size(s_charucoBoardSquareCountX, s_charucoBoardSquareCountY), s_charucoBoardSquareSideLengthMeters, s_charucoBoardMarkerSideLengthMeters, charucoDictionary());

  // Set some default detection parameters
  s_detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX; // Enable subpixel refinement for higher precision


  MultiViewCalibrationData data;

  data.views.resize(2); // number of views

  PerfTimer perfTimer;

  const char* imageFilenames_camera0[] = {
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481695.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481701.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481709.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481736.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481769.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481777.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481782.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481788.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481811.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481817.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481823.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481835.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481850.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481856.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481862.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481868.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481882.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481887.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481892.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481898.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481903.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481913.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481919.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481925.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481927.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481934.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481941.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481949.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481952.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751481958.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751482024.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751482038.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751482050.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera0_1751482057.png",
  };

  const char* imageFilenames_camera1[] = {
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481695.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481701.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481709.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481736.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481769.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481777.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481782.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481788.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481811.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481817.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481823.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481835.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481850.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481856.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481862.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481868.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481882.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481887.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481892.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481898.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481903.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481913.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481919.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481925.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481927.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481934.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481941.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481949.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481952.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751481958.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751482024.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751482038.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751482050.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera1_1751482057.png",
  };


  constexpr size_t imageCount = sizeof(imageFilenames_camera0) / sizeof(imageFilenames_camera0[0]);
  const char** viewImageFilenames[] = {imageFilenames_camera0, imageFilenames_camera1};

  for (size_t viewIdx = 0; viewIdx < data.views.size(); ++viewIdx) {
    ViewCalibrationData& viewData = data.views[viewIdx];

    viewData.observations.resize(imageCount);

    FxThreading::runArrayTask(0, imageCount, [&](size_t idx) {
      viewData.observations[idx].image = cv::imread(viewImageFilenames[viewIdx][idx], cv::IMREAD_GRAYSCALE);
    });

    // Validation: ensure all images are the same size
    assert(viewData.observations.size() >= 1);
    viewData.imageSize = cv::Size(viewData.observations[0].image.cols, viewData.observations[0].image.rows);

    for (size_t i = 1; i < viewData.observations.size(); ++i) {
      assert(viewData.imageSize.width == viewData.observations[i].image.cols && viewData.imageSize.height == viewData.observations[i].image.rows);
    }
  }

  printf("Loading complete in %.3f ms\n", perfTimer.checkpoint());

  // Cache board points
  std::vector<cv::Point3f> boardPoints = s_charucoBoard->getChessboardCorners();

  // run detections on all images
  {
    const size_t totalCorners = s_charucoBoard->getChessboardSize().width * s_charucoBoard->getChessboardSize().height;

    for (size_t viewIdx = 0; viewIdx < data.views.size(); ++viewIdx) {
      ViewCalibrationData& viewData = data.views[viewIdx];

      FxThreading::runArrayTask(0, viewData.observations.size(), [&](size_t idx) {
        Observation& obs = viewData.observations[idx];

        auto detector = createCharucoDetector();

        cv::Mat currentCharucoCorners;
        std::vector<int> currentCharucoIds;
        detector.detectBoard(obs.image, currentCharucoCorners, currentCharucoIds);

        bool found = (currentCharucoIds.size() >= (totalCorners / 3));
        if (found) {
         
          obs.objectPoints.reserve(currentCharucoIds.size()); 
          obs.imagePoints.reserve(currentCharucoIds.size());

          // Convert currentCharucoCorners mat to vector<Point2f>
          assert(currentCharucoCorners.cols == 1);
          for (size_t i = 0; i < currentCharucoCorners.rows; ++i) {
            obs.imagePoints.push_back(cv::Point2f(currentCharucoCorners.ptr<cv::Point2f>()[i]));
          }

          // Extract object points from the board definition

          for (size_t pointIdx = 0; pointIdx < currentCharucoIds.size(); ++pointIdx) {
            const cv::Point3f& boardPoint = boardPoints[currentCharucoIds[pointIdx]];
            obs.objectPoints.push_back(boardPoint);
          }
        }
        printf("[view %zu | %zu] found=%u cornerCount=%zu cornerIdCount=%zu\n", viewIdx, idx, found, currentCharucoCorners.total(), currentCharucoIds.size());
      });
    }
  }


  printf("Detection done in %.3f ms\n\n", perfTimer.checkpoint());


  printf("=== OpenCV calibration ===\n");

  // from CameraSystem::IntrinsicCalibrationContext::asyncUpdateIncrementalCalibration()
  cv::Mat stdDeviations;
  std::vector<float> reprojErrs;
  int flags =
    cv::CALIB_FIX_ASPECT_RATIO;

  // Distortion coefficients
  // k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4[, tx, tx]]]]
  int distCoeffSize = 5;

#if 0
  if (false) { // Rational model
    flags |= cv::CALIB_RATIONAL_MODEL;
    distCoeffSize = 8;
  }

  if (false) { // Thin prism model
    distCoeffSize = 12; // rational + thin_prism
    flags |= cv::CALIB_THIN_PRISM_MODEL;
  }
#endif

  for (size_t viewIdx = 0; viewIdx < data.views.size(); ++viewIdx) {
    ViewCalibrationData& viewData = data.views[viewIdx];
    printf("View %zu\n", viewIdx);

    cv::Mat intrinsicMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(distCoeffSize, 1, CV_64F);
    cv::Mat rvecs;
    cv::Mat tvecs;

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    for (size_t i = 0; i < viewData.observations.size(); ++i) {
      if (!viewData.observations[i].objectPoints.size())
        continue; // Empty observation
      objectPoints.push_back(viewData.observations[i].objectPoints);
      imagePoints.push_back(viewData.observations[i].imagePoints);
    }

    double feedbackRmsError = cv::calibrateCamera(objectPoints, imagePoints, viewData.imageSize,
                                   intrinsicMatrix, distCoeffs,
                                   rvecs, tvecs, stdDeviations, cv::noArray(),
                                   /*perViewErrors=*/ cv::noArray(), flags);

    double feedbackFovX, feedbackFovY;
    cv::Point2d feedbackPrincipalPoint;

    double focalLength, aspectRatio; // not valid without a real aperture size, which we don't bother providing
    cv::calibrationMatrixValues(intrinsicMatrix, viewData.imageSize, 0.0, 0.0, feedbackFovX, feedbackFovY, focalLength, feedbackPrincipalPoint, aspectRatio);


    printf("Calibration complete in %.3f ms\n", perfTimer.checkpoint());
    printf("RMS error: %f\n", feedbackRmsError);
    printf("FOV: %.2f x %.2f\n", feedbackFovX, feedbackFovY);

    printf("\nIntrinsic matrix:\n");
    for (size_t i = 0; i < 3; ++i) {
      printf("%.6f %.6f %.6f\n",
        intrinsicMatrix.ptr<double>(i)[0],
        intrinsicMatrix.ptr<double>(i)[1],
        intrinsicMatrix.ptr<double>(i)[2]);
    }
    printf("\nCoeffs: ");
    for (int i = 0; i < distCoeffSize; ++i) {
      printf("%.8f ", distCoeffs.ptr<double>()[i]);
    }
    printf("\n\n");
  } // View loop


  printf("=== industrial_calibration calibration ===\n");
  perfTimer.checkpoint();

  for (size_t viewIdx = 0; viewIdx < data.views.size(); ++viewIdx) {
    ViewCalibrationData& viewData = data.views[viewIdx];
    printf("View %zu\n", viewIdx);

    industrial_calibration::CameraIntrinsicProblem intrinsicProblem;

    // Initial intrinsics guess from OpenCV's cv::initCameraMatrix2D
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    for (size_t i = 0; i < viewData.observations.size(); ++i) {
      if (!viewData.observations[i].objectPoints.size())
        continue; // Empty observation
      objectPoints.push_back(viewData.observations[i].objectPoints);
      imagePoints.push_back(viewData.observations[i].imagePoints);
    }

    cv::Mat intrinsicGuess = cv::initCameraMatrix2D(/*objectPoints=*/ objectPoints, /*imagePoints=*/ imagePoints, viewData.imageSize);

    printf("\nIntrinsic initial guess:\n");
    for (size_t i = 0; i < 3; ++i) {
      printf("%.6f %.6f %.6f\n",
        intrinsicGuess.ptr<double>(i)[0],
        intrinsicGuess.ptr<double>(i)[1],
        intrinsicGuess.ptr<double>(i)[2]);
    }

    intrinsicProblem.intrinsics_guess.fx() = intrinsicGuess.ptr<double>(0)[0];
    intrinsicProblem.intrinsics_guess.fy() = intrinsicGuess.ptr<double>(1)[1];
    intrinsicProblem.intrinsics_guess.cx() = intrinsicGuess.ptr<double>(0)[2];
    intrinsicProblem.intrinsics_guess.cy() = intrinsicGuess.ptr<double>(1)[2];

    // Use extrinsic guesses from OpenCV's solvePnP (no calibration/distortion)
    intrinsicProblem.use_extrinsic_guesses = true;

    for (size_t observationIdx = 0; observationIdx < viewData.observations.size(); ++observationIdx) {
      Observation& obs = viewData.observations[observationIdx];
      if (obs.empty())
        continue; // No data for this frame on this view

      // Use OpenCV's solvePnP to populate the extrinsic guess
      cv::Mat rvec, tvec;
      cv::solvePnP(/*objectPoints=*/ obs.objectPoints, /*imagePoints=*/ obs.imagePoints, /*cameraMatrix=*/ intrinsicGuess, /*distCoeffs=*/ cv::Mat(), rvec, tvec, /*useExtrinsicGuess=*/ false);

      // Convert to rotation matrix
      cv::Mat rotation(3, 3, CV_64F);
      cv::Rodrigues(rvec, rotation);

      Eigen::Isometry3d pose;
      pose.linear() = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Map(reinterpret_cast<double*>(rotation.data));
      pose.translation() = Eigen::Vector3d::Map(tvec.ptr<double>());

      intrinsicProblem.extrinsic_guesses.push_back(pose);
      intrinsicProblem.image_observations.push_back(obs.correspondenceSet());
    }

    industrial_calibration::CameraIntrinsicResult res = industrial_calibration::optimize(intrinsicProblem);
    std::cout << res << std::endl;
    
    printf("Calibration complete in %.3f ms\n", perfTimer.checkpoint());

  } // View loop




  FxThreading::detail::shutdown();

  return 0;
}

