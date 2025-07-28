#include <cstdio>
#include <cstdint>
#include <cassert>
#include <vector>
#include <iostream>

#include "common/Timing.h"

#include <industrial_calibration/optimizations/camera_intrinsic.h>
#include <industrial_calibration/optimizations/extrinsic_multi_static_camera.h>
#include <ceres/ceres.h>
#include <unsupported/Eigen/EulerAngles>

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

void printIsometry(const char* prefix, const Eigen::Isometry3d& xf, const char* postfix = "") {
  auto tx = xf.translation();
  auto rx = Eigen::EulerAnglesXYZd(xf.rotation()).angles();
  const double rad2deg = 180.0 / M_PI;
  printf("%stx=[%.6f, %.6f, %.6f] rx=[%.6f, %.6f, %.6f]%s",
    prefix, tx[0], tx[1], tx[2], rx[0] * rad2deg, rx[1] * rad2deg, rx[2] * rad2deg, postfix);
}
void printIsometry(const Eigen::Isometry3d& xf, const char* postfix = "") { printIsometry("", xf, postfix); }


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

  std::vector<int> objectPointIds; // IDs of points on the calibration target
  std::vector<cv::Point3f> objectPoints; // Points in object space, on the calibration target
  std::vector<cv::Point2f> imagePoints; // Points in image space

  Eigen::Isometry3d targetTransform = Eigen::Isometry3d::Identity(); // Extrinsic / target transform, recorded during intrinsic calibration.
};

struct ViewCalibrationData {
  cv::Size imageSize;
  std::vector<Observation> observations;
  industrial_calibration::CameraIntrinsicResult intrinsicCalibration;

  cv::Mat cvIntrinsicMatrix() {
    // [fx  0 cx ]
    // [ 0 fy cy ]
    // [ 0  0  1 ]
    cv::Mat res = cv::Mat::eye(3, 3, CV_64F);
    res.ptr<double>(0)[0] = intrinsicCalibration.intrinsics.fx();
    res.ptr<double>(1)[1] = intrinsicCalibration.intrinsics.fy();
    res.ptr<double>(0)[2] = intrinsicCalibration.intrinsics.cx();
    res.ptr<double>(1)[2] = intrinsicCalibration.intrinsics.cy();
    return res;
  }

  cv::Mat cvDistCoeffs() {
    cv::Mat res = cv::Mat::zeros(5, 1, CV_64F);
    double* pd = res.ptr<double>();
    pd[0] = intrinsicCalibration.distortions[0]; // k1
    pd[1] = intrinsicCalibration.distortions[1]; // k2
    pd[2] = intrinsicCalibration.distortions[2]; // p1
    pd[3] = intrinsicCalibration.distortions[3]; // p2
    pd[4] = intrinsicCalibration.distortions[4]; // k3
    return res;
  }

};


struct MultiViewCalibrationData {

  size_t observationCount() {
    // All views should have the same number of observations
    return views[0].observations.size();
  }

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

  data.views.resize(4); // number of cameras/views

  PerfTimer perfTimer;

  const char* imageFilenames[] = {
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481695.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481701.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481709.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481736.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481769.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481777.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481782.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481788.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481811.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481817.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481823.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481835.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481850.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481856.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481862.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481868.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481882.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481887.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481892.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481898.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481903.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481913.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481919.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481925.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481927.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481934.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481941.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481949.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481952.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481958.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482024.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482038.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482050.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482057.png",
  };

  constexpr size_t imageCount = sizeof(imageFilenames) / sizeof(imageFilenames[0]);

  for (size_t viewIdx = 0; viewIdx < data.views.size(); ++viewIdx) {
    ViewCalibrationData& viewData = data.views[viewIdx];

    viewData.observations.resize(imageCount);

    FxThreading::runArrayTask(0, imageCount, [&](size_t idx) {
      char buf[512];
      snprintf(buf, 512, imageFilenames[idx], (unsigned int) viewIdx);
      viewData.observations[idx].image = cv::imread(buf, cv::IMREAD_GRAYSCALE);
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

          // Save point ids
          obs.objectPointIds = std::move(currentCharucoIds);
        }
        printf("[view %zu | %zu] found=%u cornerCount=%zu cornerIdCount=%zu\n", viewIdx, idx, found, currentCharucoCorners.total(), currentCharucoIds.size());
      });
    }
  }


  printf("Detection done in %.3f ms\n\n", perfTimer.checkpoint());

#if 0
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

#endif // OpenCV calibration


  printf("=== industrial_calibration intrinsic calibration ===\n");
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


    std::vector<size_t> targetTransformToObservationIdx;

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

      targetTransformToObservationIdx.push_back(observationIdx);
    }

    viewData.intrinsicCalibration = industrial_calibration::optimize(intrinsicProblem);
    std::cout << viewData.intrinsicCalibration << std::endl;

    for (size_t i = 0; i < intrinsicProblem.extrinsic_guesses.size(); ++i) {
      printf("Extrinsics [%zu]:", i);
      printIsometry(" Guess = ", intrinsicProblem.extrinsic_guesses[i]);
      printIsometry(" Actual = ", viewData.intrinsicCalibration.target_transforms[i], "\n");
    }

    for (size_t i = 0; i < targetTransformToObservationIdx.size(); ++i) {
      viewData.observations[targetTransformToObservationIdx[i]].targetTransform = viewData.intrinsicCalibration.target_transforms[i];
    }
    
    printf("Calibration complete in %.3f ms\n", perfTimer.checkpoint());

  } // View loop


  // Guesses for the pose between base (view 0) and other views
  std::vector<Eigen::Isometry3d> base_to_camera_guess(4, Eigen::Isometry3d::Identity());

#if 1

  printf("=== OpenCV stereo calibration ===\n");
  // "Left" and "Right" here are arbitrary -- we just try to calibrate between every view-pair.
  for (size_t leftViewIdx = 0; leftViewIdx < data.views.size(); ++leftViewIdx) {
    ViewCalibrationData& leftViewData = data.views[leftViewIdx];

    for (size_t rightViewIdx = leftViewIdx + 1; rightViewIdx < data.views.size(); ++rightViewIdx) {
      ViewCalibrationData& rightViewData = data.views[rightViewIdx];

      // Find observations that appear in both left and right views, and collect their overlapping object and image points.
      // Object points will be identical across all same-index observations.
      std::vector<std::vector<cv::Point3f> > objectPoints;
      std::vector<std::vector<cv::Point2f> > leftImagePoints, rightImagePoints;

      assert(leftViewData.observations.size() == rightViewData.observations.size());
      for (size_t observationIdx = 0; observationIdx < leftViewData.observations.size(); ++observationIdx) {
        Observation& leftObs = leftViewData.observations[observationIdx];
        Observation& rightObs = rightViewData.observations[observationIdx];
        if (leftObs.empty() || rightObs.empty())
          continue; // No overlap for this observation

        std::vector<cv::Point3f> obsObjectPoints;
        std::vector<cv::Point2f> obsLeftImagePoints, obsRightImagePoints;

        // For each left observed point, find the corresponding right observed point
        // Terrible O(N^2) code
        for (size_t pointIdx = 0; pointIdx < leftObs.pointCount(); ++pointIdx) {
          int leftPointId = leftObs.objectPointIds[pointIdx];

          for (size_t rightPointIdx = 0; rightPointIdx < rightObs.pointCount(); ++rightPointIdx) {
            if (rightObs.objectPointIds[rightPointIdx] == leftPointId) {
              // Match
              obsObjectPoints.push_back(leftObs.objectPoints[pointIdx]);
              obsLeftImagePoints.push_back(leftObs.imagePoints[pointIdx]);
              obsRightImagePoints.push_back(rightObs.imagePoints[rightPointIdx]);
              break;
            }
          }
        }

        // Need at least 6 points to line up an observation
        if (obsObjectPoints.size() >= 6) {
          objectPoints.push_back(std::move(obsObjectPoints));
          leftImagePoints.push_back(std::move(obsLeftImagePoints));
          rightImagePoints.push_back(std::move(obsRightImagePoints));
        }
      }


      cv::Mat stereoRotation, stereoTranslation;
      cv::Mat rvecs, tvecs, E, F;

      printf("cv::stereoCalibrate() view %zu to %zu\n", leftViewIdx, rightViewIdx);
      printf("  %zu observations\n", objectPoints.size());

      double reprojectionError = cv::stereoCalibrate(
        /*objectPoints =  */ objectPoints,
        /*imagePoints1 =  */ leftImagePoints,
        /*imagePoints2 =  */ rightImagePoints,
        /*cameraMatrix1 = */ leftViewData.cvIntrinsicMatrix(),
        /*distCoeffs1 =   */ leftViewData.cvDistCoeffs(),
        /*cameraMatrix1 = */ rightViewData.cvIntrinsicMatrix(),
        /*distCoeffs1 =   */ rightViewData.cvDistCoeffs(),
        /*imageSize =     */ leftViewData.imageSize,
        /*R = */ stereoRotation,
        /*T = */ stereoTranslation,
        /*E = */ E,
        /*F = */ F,
        /*rvecs = */ rvecs,
        /*tvecs = */ tvecs,
        /*perViewErrors = */ cv::noArray(),
        /*flags = */ cv::CALIB_FIX_INTRINSIC
      );

      //std::cout << stereoTranslation << std::endl << stereoRotation << std::endl;

      // Covert R and T to isometry

      Eigen::Isometry3d stereoOffset;
      {
        assert(stereoTranslation.total() == 3 && stereoTranslation.type() == CV_64F);
        // stereoCalibrate should return a 3x3 matrix in stereoRotation, not an rvec.
        assert(stereoRotation.total() == 9 && stereoRotation.type() == CV_64F);

        stereoOffset.linear() = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Map(stereoRotation.ptr<double>());
        stereoOffset.translation() = Eigen::Vector3d::Map(stereoTranslation.ptr<double>());
      }

      printIsometry("  Stereo offset: ", stereoOffset, "\n");

      printf("  Reprojection error: %.6f\n", reprojectionError);

      // Update base-to-camera guesses
      if (leftViewIdx == 0) {
        base_to_camera_guess[rightViewIdx] = stereoOffset;
      }
    }
  }


#endif // OpenCV stereo calibration

  // Multi-view
  printf("=== industrial_calibration multi-view calibration ===\n");
  {
    industrial_calibration::ExtrinsicMultiStaticCameraOnlyProblem problem;

    problem.fix_first_camera = true;

    // Per-view intrinsics
    for (size_t viewIdx = 0; viewIdx < data.views.size(); ++viewIdx) {
      ViewCalibrationData& viewData = data.views[viewIdx];
      problem.intr.push_back(viewData.intrinsicCalibration.intrinsics);
    }

    // Sanity check on observation counts
    for (size_t viewIdx = 1; viewIdx < data.views.size(); ++viewIdx) {
      assert(data.views[viewIdx].observations.size() == data.views[0].observations.size());
    }

    // Per-view base-to-camera guesses -- use cv::stereoCalibrate guesses
#if 0
    problem.base_to_camera_guess = base_to_camera_guess;
#else
    problem.base_to_camera_guess = std::vector(data.views.size(), Eigen::Isometry3d::Identity());
#endif

    // Per-observation base-to-target guesses.
    // We initialize this with the target transforms from the intrinsic calibration process.
    // Not all target transforms will be available for the base view (view 0), though, so we use other views
    // and compose their initial base-to-camera guesses to come up with base-to-target guesses.

    problem.base_to_target_guess.resize(data.observationCount());
    for (size_t i = 0; i < problem.base_to_target_guess.size(); ++i) {
      problem.base_to_target_guess[i] = data.views[0].observations[i].targetTransform;

      bool isValid = !data.views[0].observations[i].empty();
      size_t pointCount = data.views[0].observations[i].pointCount();

      printIsometry("Base-to-target guess from view 0: ", problem.base_to_target_guess[i], "\n");

      for (size_t viewIdx = 1; viewIdx < data.views.size(); ++viewIdx) {
        if (data.views[viewIdx].observations[i].targetTransform.isApprox(Eigen::Isometry3d::Identity()))
          continue;

        printf("  Base-to-target guess transformed from view %zu: ", viewIdx);
        Eigen::Isometry3d xf = base_to_camera_guess[viewIdx] * data.views[viewIdx].observations[i].targetTransform;

        printIsometry(xf, "\n");

        if (!isValid) {
          // For observations that don't have a valid base transform, compose a guess transform from the other
          // view that saw the most points.
          if (data.views[viewIdx].observations[i].pointCount() > pointCount) {
            pointCount = data.views[viewIdx].observations[i].pointCount();
            problem.base_to_target_guess[i] = xf;
          }
        }
      }
    }

    // Set up image observations
    problem.image_observations.resize(data.views.size());
    for (size_t viewIdx = 0; viewIdx < data.views.size(); ++viewIdx) {
      ViewCalibrationData& viewData = data.views[viewIdx];
      std::vector<industrial_calibration::Correspondence2D3D::Set>& viewObsSets = problem.image_observations[viewIdx];
      viewObsSets.resize(viewData.observations.size());

      for (size_t obsIdx = 0; obsIdx < viewData.observations.size(); ++obsIdx) {
        viewObsSets[obsIdx] = viewData.observations[obsIdx].correspondenceSet();

      }
    }

    // Run solver
    industrial_calibration::ExtrinsicMultiStaticCameraOnlyResult result = optimize(problem);

    std::cout << "Optimization " << (result.converged ? "converged" : "did not converge") << "\n"
           << "Initial cost per observation (pixels): " << std::sqrt(result.initial_cost_per_obs) << "\n"
           << "Final cost per observation (pixels): " << std::sqrt(result.final_cost_per_obs) << "\n\n";

    if (result.converged) {
      for (size_t i = 0; i < result.base_to_target.size(); ++i) {
        printf("Base-to-target %zu: ", i);
        printIsometry(result.base_to_target[i], "\n");
      }
      for (size_t i = 0; i < result.base_to_camera.size(); ++i) {
        printf("Base-to-camera %zu: ", i);
        printIsometry(result.base_to_camera[i], "\n");
      }
    }
  }



  FxThreading::detail::shutdown();

  return 0;
}

