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














int main(int argc, char** argv) {
  FxThreading::detail::init();

  // Initialize ChAruCo data on first use
  if (!s_charucoBoard)
    s_charucoBoard = new cv::aruco::CharucoBoard(cv::Size(s_charucoBoardSquareCountX, s_charucoBoardSquareCountY), s_charucoBoardSquareSideLengthMeters, s_charucoBoardMarkerSideLengthMeters, charucoDictionary());

  // Set some default detection parameters
  s_detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX; // Enable subpixel refinement for higher precision

  std::vector<std::vector<cv::Point2f> > m_allCharucoCorners;
  std::vector<std::vector<int> > m_allCharucoIds;
  cv::Mat m_perViewErrors;
  double m_feedbackRmsError;
  double m_feedbackFovX, m_feedbackFovY; // degrees
  cv::Point2d m_feedbackPrincipalPoint;


  std::vector<cv::Mat> images;
  cv::Size imageSize;

  PerfTimer perfTimer;

  const char* imageFilenames[] = {
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
#if 1 // Cut off rest of samples for faster testing
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
#endif
  };


  constexpr size_t imageCount = sizeof(imageFilenames) / sizeof(imageFilenames[0]);
  images.resize(imageCount);

  FxThreading::runArrayTask(0, imageCount, [&](size_t idx) {
    images[idx] = cv::imread(imageFilenames[idx], cv::IMREAD_GRAYSCALE);
  });

  // Validation: ensure all images are the same size
  assert(images.size() >= 1);
  imageSize = cv::Size(images[0].cols, images[0].rows);

  for (size_t i = 1; i < images.size(); ++i) {
    assert(imageSize.width == images[i].cols && imageSize.height == images[i].rows);
  }

  printf("Loading complete in %.3f ms\n", perfTimer.checkpoint());

  // run detections on all images
  {
    // Protect simultaneous writes to the detection data arrays
    std::mutex dataMutex;

    FxThreading::runArrayTask(0, imageCount, [&](size_t idx) {

      auto detector = createCharucoDetector();

      cv::Mat currentCharucoCorners;
      std::vector<int> currentCharucoIds;
      detector.detectBoard(images[idx], currentCharucoCorners, currentCharucoIds);

      size_t totalCorners = s_charucoBoard->getChessboardSize().width * s_charucoBoard->getChessboardSize().height;
      bool found = (currentCharucoIds.size() >= (totalCorners / 3));
      if (found) {

        std::vector<cv::Point2f> corners;

        // Convert currentCharucoCorners mat to vector<Point2f>
        assert(currentCharucoCorners.cols == 1);
        for (size_t i = 0; i < currentCharucoCorners.rows; ++i) {
          corners.push_back(cv::Point2f(currentCharucoCorners.ptr<cv::Point2f>()[i]));
        }

        std::lock_guard<std::mutex> dataLock(dataMutex);

        m_allCharucoCorners.push_back(std::move(corners));
        m_allCharucoIds.push_back(currentCharucoIds);
      }
      printf("[%zu] found=%u cornerCount=%zu cornerIdCount=%zu\n", idx, found, currentCharucoCorners.total(), currentCharucoIds.size());

    });
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

  if (false) { // Rational model
    flags |= cv::CALIB_RATIONAL_MODEL;
    distCoeffSize = 8;
  }

  if (false) { // Thin prism model
    distCoeffSize = 12; // rational + thin_prism
    flags |= cv::CALIB_THIN_PRISM_MODEL;
  }

  cv::Mat intrinsicMatrix = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat distCoeffs = cv::Mat::zeros(distCoeffSize, 1, CV_64F);
  cv::Mat rvecs;
  cv::Mat tvecs;

  m_feedbackRmsError = cv::aruco::calibrateCameraCharuco(m_allCharucoCorners, m_allCharucoIds,
                                 s_charucoBoard, imageSize,
                                 intrinsicMatrix, distCoeffs,
                                 rvecs, tvecs, stdDeviations, cv::noArray(),
                                 m_perViewErrors, flags);

  double focalLength, aspectRatio; // not valid without a real aperture size, which we don't bother providing
  cv::calibrationMatrixValues(intrinsicMatrix, imageSize, 0.0, 0.0, m_feedbackFovX, m_feedbackFovY, focalLength, m_feedbackPrincipalPoint, aspectRatio);


  printf("Calibration complete in %.3f ms\n", perfTimer.checkpoint());
  printf("RMS error: %f\n", m_feedbackRmsError);
  printf("FOV: %.2f x %.2f\n", m_feedbackFovX, m_feedbackFovY);

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


  printf("=== industrial_calibration calibration ===\n");
  perfTimer.checkpoint();

  // Cache board points
  std::vector<cv::Point3f> boardPoints = s_charucoBoard->getChessboardCorners();


  industrial_calibration::CameraIntrinsicProblem intrinsicProblem;

  // Initial intrinsics guess from OpenCV's cv::initCameraMatrix2D
  std::vector<std::vector<cv::Point3f> > objectPoints;
  objectPoints.resize(m_allCharucoCorners.size());

  for (size_t observationIdx = 0; observationIdx < m_allCharucoCorners.size(); ++observationIdx) {
    const std::vector<int>& pointIds = m_allCharucoIds[observationIdx];

    for (size_t pointIdx = 0; pointIdx < pointIds.size(); ++pointIdx) {
      const cv::Point3f& boardPoint = boardPoints[pointIds[pointIdx]];
      objectPoints[observationIdx].push_back(boardPoint);
    }
  }

  cv::Mat intrinsicGuess = cv::initCameraMatrix2D(/*objectPoints=*/ objectPoints, /*imagePoints=*/ m_allCharucoCorners, imageSize);

  printf("\nIntrinsic initial guess:\n");
  for (size_t i = 0; i < 3; ++i) {
    printf("%.6f %.6f %.6f\n",
      intrinsicGuess.ptr<double>(i)[0],
      intrinsicGuess.ptr<double>(i)[1],
      intrinsicGuess.ptr<double>(i)[2]);
  }

#if 1
  intrinsicProblem.intrinsics_guess.fx() = intrinsicGuess.ptr<double>(0)[0];
  intrinsicProblem.intrinsics_guess.fy() = intrinsicGuess.ptr<double>(1)[1];
  intrinsicProblem.intrinsics_guess.cx() = intrinsicGuess.ptr<double>(0)[2];
  intrinsicProblem.intrinsics_guess.cy() = intrinsicGuess.ptr<double>(1)[2];

#else
  // Initial intrinsics guess is taken from the OpenCV solution
  intrinsicProblem.intrinsics_guess.fx() = intrinsicMatrix.ptr<double>(0)[0];
  intrinsicProblem.intrinsics_guess.fy() = intrinsicMatrix.ptr<double>(1)[1];
  intrinsicProblem.intrinsics_guess.cx() = intrinsicMatrix.ptr<double>(0)[2];
  intrinsicProblem.intrinsics_guess.cy() = intrinsicMatrix.ptr<double>(1)[2];
#endif

#if 1
  // Use extrinsic guesses from OpenCV's solvePnP (no calibration/distortion)

  intrinsicProblem.use_extrinsic_guesses = true;
  intrinsicProblem.extrinsic_guesses.resize(m_allCharucoCorners.size());
  for (size_t observationIdx = 0; observationIdx < m_allCharucoCorners.size(); ++observationIdx) {
    cv::Mat rvec, tvec;
    cv::solvePnP(/*objectPoints=*/ objectPoints[observationIdx], /*imagePoints=*/ m_allCharucoCorners[observationIdx], /*cameraMatrix=*/ intrinsicGuess, /*distCoeffs=*/ cv::Mat(), rvec, tvec, /*useExtrinsicGuess=*/ false);

    // Convert to rotation matrix
    cv::Mat rotation(3, 3, CV_64F);
    cv::Rodrigues(rvec, rotation);

    Eigen::Isometry3d pose;
    pose.linear() = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Map(reinterpret_cast<double*>(rotation.data));
    pose.translation() = Eigen::Vector3d::Map(tvec.ptr<double>());

    intrinsicProblem.extrinsic_guesses[observationIdx] = (pose);
  }

#else
  // Use extrinsic guesses from OpenCV calibration
  intrinsicProblem.use_extrinsic_guesses = true;
  intrinsicProblem.extrinsic_guesses.resize(m_allCharucoCorners.size());
  for (size_t observationIdx = 0; observationIdx < m_allCharucoCorners.size(); ++observationIdx) {
    // rvecs and tvecs are each #observations rows x 1 column, CV_64FC3

    // Convert to rotation matrix
    cv::Mat rotation(3, 3, CV_64F);
    cv::Rodrigues(rvecs.row(observationIdx), rotation);

    Eigen::Isometry3d pose;
    pose.linear() = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Map(reinterpret_cast<double*>(rotation.data));
    pose.translation() = Eigen::Vector3d::Map(tvecs.ptr<double>(observationIdx));

    intrinsicProblem.extrinsic_guesses[observationIdx] = (pose);
  }
#endif

  // Convert observations from opencv mats to vector< vector<Correspondence2D3D> >
  intrinsicProblem.image_observations.resize(m_allCharucoCorners.size());
  for (size_t observationIdx = 0; observationIdx < m_allCharucoCorners.size(); ++observationIdx) {

    const std::vector<cv::Point2f>& imagePoints = m_allCharucoCorners[observationIdx];
    const std::vector<int>& pointIds = m_allCharucoIds[observationIdx];
    assert(imagePoints.size() == pointIds.size());

    size_t pointCount = pointIds.size();

    industrial_calibration::Correspondence2D3D::Set& observation = intrinsicProblem.image_observations[observationIdx];
    observation.resize(pointCount);
    for (size_t pointIdx = 0; pointIdx < pointCount; ++pointIdx) {

      industrial_calibration::Correspondence2D3D& corr = observation[pointIdx];
      // corr.in_image is Eigen::Matrix<double, 2, 1>
      corr.in_image = Eigen::Vector2d(imagePoints[pointIdx].x, imagePoints[pointIdx].y);

      const cv::Point3f& boardPoint = boardPoints[pointIds[pointIdx]];
      corr.in_target = Eigen::Vector3d(boardPoint.x, boardPoint.y, boardPoint.z);
    }
  }

  industrial_calibration::CameraIntrinsicResult res = industrial_calibration::optimize(intrinsicProblem);
  std::cout << res << std::endl;
  
  printf("Calibration complete in %.3f ms\n", perfTimer.checkpoint());





  FxThreading::detail::shutdown();

  return 0;
}

