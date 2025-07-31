#include <cstdio>
#include <cstdint>
#include <cassert>
#include <set>
#include <vector>
#include <iostream>
#include <iterator>
#include <algorithm>

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
  auto rx = Eigen::EulerAnglesYXZd(xf.rotation()).angles();
  const double rad2deg = 180.0 / M_PI;
  printf("%stx=[%.6f, %.6f, %.6f] rx=[%.6f, %.6f, %.6f]%s",
    prefix, tx[0], tx[1], tx[2], rx[0] * rad2deg, rx[1] * rad2deg, rx[2] * rad2deg, postfix);
}
void printIsometry(const Eigen::Isometry3d& xf, const char* postfix = "") { printIsometry("", xf, postfix); }

void read(const cv::FileNode& node, Eigen::Isometry3d& value, Eigen::Isometry3d ignored_default_value = Eigen::Isometry3d::Identity()) {
  // Serialize as matrix
  cv::Mat tmpMat;
  read(node, tmpMat, cv::Mat());
  assert(tmpMat.total() == 16 && tmpMat.type() == CV_64F);
  double* dest = value.matrix().data();
  const double* src = tmpMat.ptr<double>();
  for (size_t i = 0; i < 16; ++i)
    dest[i] = src[i];
}

void read(const cv::FileNode& node, std::vector<Eigen::Isometry3d>& value) {
  assert(node.isSeq());
  value.resize(node.size());
  for (size_t i = 0; i < value.size(); ++i) {
    read(node[i], value[i]);
  }
}


void write(cv::FileStorage& fs, const cv::String& name, const Eigen::Isometry3d& value) {
  // Serialize as matrix
  cv::Matx<double, 4, 4> mat(const_cast<double*>(value.matrix().data()));
  write(fs, name, mat);
}

void write(cv::FileStorage& fs, const cv::String& name, const std::vector<Eigen::Isometry3d>& value) {
  fs.startWriteStruct(name, cv::FileNode::SEQ, cv::String());
  for (size_t i = 0; i < value.size(); ++i) {
    write(fs, cv::String(), value[i]);
  }
  fs.endWriteStruct();
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

  std::vector<int> objectPointIds; // IDs of points on the calibration target
  std::vector<cv::Point3f> objectPoints; // Points in object space, on the calibration target
  std::vector<cv::Point2f> imagePoints; // Points in image space

  Eigen::Isometry3d targetTransform = Eigen::Isometry3d::Identity(); // Extrinsic / target transform, recorded during intrinsic calibration.



  void loadCalibrationData(cv::FileNode fn) {
    if (!fn.isMap())
      throw std::runtime_error("Unexpected type for Observation node (not Map)");

    fn["objectPointIds"] >> objectPointIds;
    fn["objectPoints"] >> objectPoints;
    fn["imagePoints"] >> imagePoints;
    read(fn["targetTransform"], targetTransform);
  }

  void saveCalibrationData(cv::FileStorage& fs) {
    fs.startWriteStruct(cv::String(), cv::FileNode::MAP, cv::String());

    write(fs, "objectPointIds", objectPointIds);
    write(fs, "objectPoints", objectPoints);
    write(fs, "imagePoints", imagePoints);
    write(fs, "targetTransform", targetTransform);

    fs.endWriteStruct();
  }
};

struct CameraCalibrationData {
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


  void loadCalibrationData(cv::FileNode fn) {
    if (!fn.isMap())
      throw std::runtime_error("Unexpected type for camera node (not Map)");

    cv::Mat tmpMat;

    fn["imageSize"] >> imageSize;
    fn["converged"] >> intrinsicCalibration.converged;
    fn["initial_cost_per_obs"] >> intrinsicCalibration.initial_cost_per_obs;
    fn["final_cost_per_obs"] >> intrinsicCalibration.final_cost_per_obs;

    fn["intrinsicMatrix"] >> tmpMat;
    intrinsicCalibration.intrinsics.fx() = tmpMat.ptr<double>(0)[0];
    intrinsicCalibration.intrinsics.aspect() = tmpMat.ptr<double>(1)[1] / tmpMat.ptr<double>(0)[0];
    intrinsicCalibration.intrinsics.cx() = tmpMat.ptr<double>(0)[2];
    intrinsicCalibration.intrinsics.cy() = tmpMat.ptr<double>(1)[2];

    fn["distortionCoeffs"] >> tmpMat;
    for (size_t i = 0; i < intrinsicCalibration.distortions.size(); ++i) {
      intrinsicCalibration.distortions[i] = tmpMat.ptr<double>()[i];
    }

    read(fn["targetTransforms"], intrinsicCalibration.target_transforms);

    cv::FileNode obsFn = fn["observations"];
    if (!obsFn.isSeq())
      throw std::runtime_error("Unexpected type for observations node (not Sequence)");

    observations.resize(obsFn.size());
    for (size_t i = 0; i < obsFn.size(); ++i) {
      cv::FileNode obsNode = obsFn[i];
      observations[i].loadCalibrationData(obsNode);
    }
  }

  void saveCalibrationData(cv::FileStorage& fs) {
    fs.startWriteStruct(cv::String(), cv::FileNode::MAP, cv::String());
    {
      write(fs, "imageSize", imageSize);
      write(fs, "converged", intrinsicCalibration.converged);
      write(fs, "initial_cost_per_obs", intrinsicCalibration.initial_cost_per_obs);
      write(fs, "final_cost_per_obs", intrinsicCalibration.final_cost_per_obs);

      fs.write("intrinsicMatrix", cvIntrinsicMatrix());
      fs.write("distortionCoeffs", cvDistCoeffs());
      write(fs, "targetTransforms", intrinsicCalibration.target_transforms);

      fs.startWriteStruct("observations", cv::FileNode::SEQ, cv::String());
      for (size_t i = 0; i < observations.size(); ++i) {
        observations[i].saveCalibrationData(fs);
      }
      fs.endWriteStruct();
    }
    fs.endWriteStruct();
  }

};

struct ViewCalibrationData {
  unsigned int cameraIndices[2];

  cv::Mat stereoRotation, stereoTranslation;
  cv::Mat stereoRectification[2], stereoProjection[2]; // Derived from stereoRotation/stereoTranslation via cv::stereoRectify
  cv::Mat stereoDisparityToDepth;
  cv::Rect stereoValidROI[2];
};


struct MultiCameraCalibrationData {

  size_t observationCount() {
    // All cameras should have the same number of observations
    return cameras[0].observations.size();
  }

  std::vector<CameraCalibrationData> cameras;
  std::vector<ViewCalibrationData> views;


  bool loadCalibrationData() {
    cv::FileStorage fs("multiCameraCalibrationData.yml", cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
    if (!fs.isOpened()) {
      printf("Unable to open calibration data file\n");
      return false;
    }

    try {
      cv::FileNode camerasFn = fs["cameras"];
      if (!camerasFn.isSeq())
        throw std::runtime_error("Unexpected type for root cameras node (not Sequence)");
      cameras.resize(camerasFn.size());
      for (size_t cameraIdx = 0; cameraIdx < cameras.size(); ++cameraIdx) {
        cameras[cameraIdx].loadCalibrationData(camerasFn[cameraIdx]);
      }

    } catch (const std::exception& ex) {
      printf("Unable to load calibration data: %s\n", ex.what());
      return false;
    }

    return true;
  }

  bool saveCalibrationData() {
    cv::FileStorage fs("multiCameraCalibrationData.yml", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!fs.isOpened()) {
      printf("Unable to open calibration data file\n");
      return false;
    }

    fs.startWriteStruct("cameras", cv::FileNode::SEQ, cv::String());
    for (size_t cameraIdx = 0; cameraIdx < cameras.size(); ++cameraIdx) {
      cameras[cameraIdx].saveCalibrationData(fs);
    }
    fs.endWriteStruct();

    return true;
  }


};



void generateMultiCameraCalibrationData(MultiCameraCalibrationData& data) {
  PerfTimer perfTimer;

  data.cameras.resize(4); // number of cameras


  const char* imageFilenames[] = {
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481608.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481616.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481622.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481627.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481632.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481641.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481658.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481661.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481663.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481666.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481671.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481684.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751481687.png",
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
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482063.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482071.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482079.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482263.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482271.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482276.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482283.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482288.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482293.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482298.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482303.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482309.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482317.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482323.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482330.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482342.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482349.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482357.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482362.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482368.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482374.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482379.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482385.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482391.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482397.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482403.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482410.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482417.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482425.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482443.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482454.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482460.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482467.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482473.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482480.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482487.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482493.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482499.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482505.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482511.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482516.png",
    "/home/dweatherford/calibrationData/20250702_1356/camera%u_1751482523.png",
  };

  constexpr size_t imageCount = sizeof(imageFilenames) / sizeof(imageFilenames[0]);

  for (size_t cameraIdx = 0; cameraIdx < data.cameras.size(); ++cameraIdx) {
    CameraCalibrationData& cameraData = data.cameras[cameraIdx];

    cameraData.observations.resize(imageCount);

    FxThreading::runArrayTask(0, imageCount, [&](size_t idx) {
      char buf[512];
      snprintf(buf, 512, imageFilenames[idx], (unsigned int) cameraIdx);
      cameraData.observations[idx].image = cv::imread(buf, cv::IMREAD_GRAYSCALE);
    });

    // Validation: ensure all images are the same size
    assert(cameraData.observations.size() >= 1);
    cameraData.imageSize = cv::Size(cameraData.observations[0].image.cols, cameraData.observations[0].image.rows);

    for (size_t i = 1; i < cameraData.observations.size(); ++i) {
      assert(cameraData.imageSize.width == cameraData.observations[i].image.cols && cameraData.imageSize.height == cameraData.observations[i].image.rows);
    }
  }

  printf("Loading complete in %.3f ms\n", perfTimer.checkpoint());

  // Cache board points
  std::vector<cv::Point3f> boardPoints = s_charucoBoard->getChessboardCorners();

  // run detections on all images
  {
    const size_t totalCorners = s_charucoBoard->getChessboardSize().width * s_charucoBoard->getChessboardSize().height;

    for (size_t cameraIdx = 0; cameraIdx < data.cameras.size(); ++cameraIdx) {
      CameraCalibrationData& cameraData = data.cameras[cameraIdx];

      FxThreading::runArrayTask(0, cameraData.observations.size(), [&](size_t idx) {
        Observation& obs = cameraData.observations[idx];

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
        printf("[Camera %zu | %zu] found=%u cornerCount=%zu\n", cameraIdx, idx, found, currentCharucoCorners.total());
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

  for (size_t cameraIdx = 0; cameraIdx < data.cameras.size(); ++cameraIdx) {
    CameraCalibrationData& cameraData = data.cameras[cameraIdx];
    printf("Camera %zu\n", cameraIdx);

    cv::Mat intrinsicMatrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat distCoeffs = cv::Mat::zeros(distCoeffSize, 1, CV_64F);
    cv::Mat rvecs;
    cv::Mat tvecs;

    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    for (size_t i = 0; i < cameraData.observations.size(); ++i) {
      if (!cameraData.observations[i].objectPoints.size())
        continue; // Empty observation
      objectPoints.push_back(cameraData.observations[i].objectPoints);
      imagePoints.push_back(cameraData.observations[i].imagePoints);
    }

    double feedbackRmsError = cv::calibrateCamera(objectPoints, imagePoints, cameraData.imageSize,
                                   intrinsicMatrix, distCoeffs,
                                   rvecs, tvecs, stdDeviations, cv::noArray(),
                                   /*perViewErrors=*/ cv::noArray(), flags);

    double feedbackFovX, feedbackFovY;
    cv::Point2d feedbackPrincipalPoint;

    double focalLength, aspectRatio; // not valid without a real aperture size, which we don't bother providing
    cv::calibrationMatrixValues(intrinsicMatrix, cameraData.imageSize, 0.0, 0.0, feedbackFovX, feedbackFovY, focalLength, feedbackPrincipalPoint, aspectRatio);


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
  } // Camera loop

#endif // OpenCV calibration


  printf("=== industrial_calibration intrinsic calibration ===\n");
  perfTimer.checkpoint();

  for (size_t cameraIdx = 0; cameraIdx < data.cameras.size(); ++cameraIdx) {
    CameraCalibrationData& cameraData = data.cameras[cameraIdx];
    printf("Camera %zu\n", cameraIdx);

    industrial_calibration::CameraIntrinsicProblem intrinsicProblem;

    // Initial intrinsics guess from OpenCV's cv::initCameraMatrix2D
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;
    for (size_t i = 0; i < cameraData.observations.size(); ++i) {
      if (!cameraData.observations[i].objectPoints.size())
        continue; // Empty observation
      objectPoints.push_back(cameraData.observations[i].objectPoints);
      imagePoints.push_back(cameraData.observations[i].imagePoints);
    }

    cv::Mat intrinsicGuess = cv::initCameraMatrix2D(/*objectPoints=*/ objectPoints, /*imagePoints=*/ imagePoints, cameraData.imageSize);

    printf("\nIntrinsic initial guess:\n");
    for (size_t i = 0; i < 3; ++i) {
      printf("%.6f %.6f %.6f\n",
        intrinsicGuess.ptr<double>(i)[0],
        intrinsicGuess.ptr<double>(i)[1],
        intrinsicGuess.ptr<double>(i)[2]);
    }

    intrinsicProblem.intrinsics_guess.fx() = intrinsicGuess.ptr<double>(0)[0];
    intrinsicProblem.intrinsics_guess.cx() = intrinsicGuess.ptr<double>(0)[2];
    intrinsicProblem.intrinsics_guess.cy() = intrinsicGuess.ptr<double>(1)[2];

    // Fix pixel aspect ratio to 1:1
    intrinsicProblem.intrinsics_guess.aspect() = 1.0f;
    intrinsicProblem.fix_aspect = true;

    // Use extrinsic guesses from OpenCV's solvePnP (no calibration/distortion)
    intrinsicProblem.use_extrinsic_guesses = true;


    std::vector<size_t> targetTransformToObservationIdx;

    for (size_t observationIdx = 0; observationIdx < cameraData.observations.size(); ++observationIdx) {
      Observation& obs = cameraData.observations[observationIdx];
      if (obs.empty())
        continue; // No data for this frame on this camera

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

    cameraData.intrinsicCalibration = industrial_calibration::optimize(intrinsicProblem);
    std::cout << cameraData.intrinsicCalibration << std::endl;

#if 0
    for (size_t i = 0; i < intrinsicProblem.extrinsic_guesses.size(); ++i) {
      printf("Extrinsics [%zu]:", i);
      printIsometry(" Guess = ", intrinsicProblem.extrinsic_guesses[i]);
      printIsometry(" Actual = ", cameraData.intrinsicCalibration.target_transforms[i], "\n");
    }
#endif

    for (size_t i = 0; i < targetTransformToObservationIdx.size(); ++i) {
      cameraData.observations[targetTransformToObservationIdx[i]].targetTransform = cameraData.intrinsicCalibration.target_transforms[i];
    }
    
    printf("Calibration complete in %.3f ms\n", perfTimer.checkpoint());

  } // Camera loop



  data.saveCalibrationData();
}


int main(int argc, char** argv) {
  FxThreading::detail::init();

  // Initialize ChAruCo data on first use
  if (!s_charucoBoard)
    s_charucoBoard = new cv::aruco::CharucoBoard(cv::Size(s_charucoBoardSquareCountX, s_charucoBoardSquareCountY), s_charucoBoardSquareSideLengthMeters, s_charucoBoardMarkerSideLengthMeters, charucoDictionary());

  // Set some default detection parameters
  s_detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX; // Enable subpixel refinement for higher precision


  MultiCameraCalibrationData data;

  if (data.loadCalibrationData()) {
    // Print previously-cached intrinsic calibration results
    for (size_t cameraIdx = 0; cameraIdx < data.cameras.size(); ++cameraIdx) {
      std::cout << data.cameras[cameraIdx].intrinsicCalibration << std::endl;
    }
  } else {
    printf("Couldn't load calibration data checkpoint, generating calibration from images\n");
    generateMultiCameraCalibrationData(data);
  }


  PerfTimer perfTimer;

  // Guesses for the pose between base (camera 0) and other cameras
  std::vector<Eigen::Isometry3d> base_to_camera_guess(4, Eigen::Isometry3d::Identity());

#if 1

  printf("=== OpenCV stereo calibration ===\n");
  std::vector<boost::function<void()>> taskCompletions;

  // "Left" and "Right" here are arbitrary -- we just try to calibrate between every camera-pair.
  // Launch array tasks for stereo calibration
  for (size_t leftCameraIdx = 0; leftCameraIdx < data.cameras.size(); ++leftCameraIdx) {
    taskCompletions.push_back(FxThreading::runArrayTaskAsync(/*startValue=*/ leftCameraIdx + 1, /*endValue=*/ data.cameras.size(), [leftCameraIdx, &data, &base_to_camera_guess](size_t rightCameraIdx) {

      CameraCalibrationData& leftCameraData = data.cameras[leftCameraIdx];
      CameraCalibrationData& rightCameraData = data.cameras[rightCameraIdx];

      // Find observations that appear in both left and right cameras, and collect their overlapping object and image points.
      // Object points will be identical across all same-index observations.
      std::vector<std::vector<cv::Point3f> > objectPoints;
      std::vector<std::vector<cv::Point2f> > leftImagePoints, rightImagePoints;

      assert(leftCameraData.observations.size() == rightCameraData.observations.size());
      for (size_t observationIdx = 0; observationIdx < leftCameraData.observations.size(); ++observationIdx) {
        Observation& leftObs = leftCameraData.observations[observationIdx];
        Observation& rightObs = rightCameraData.observations[observationIdx];
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

      PerfTimer perfTimer;

      double reprojectionError = cv::stereoCalibrate(
        /*objectPoints =  */ objectPoints,
        /*imagePoints1 =  */ leftImagePoints,
        /*imagePoints2 =  */ rightImagePoints,
        /*cameraMatrix1 = */ leftCameraData.cvIntrinsicMatrix(),
        /*distCoeffs1 =   */ leftCameraData.cvDistCoeffs(),
        /*cameraMatrix1 = */ rightCameraData.cvIntrinsicMatrix(),
        /*distCoeffs1 =   */ rightCameraData.cvDistCoeffs(),
        /*imageSize =     */ leftCameraData.imageSize,
        /*R = */ stereoRotation,
        /*T = */ stereoTranslation,
        /*E = */ E,
        /*F = */ F,
        /*rvecs = */ rvecs,
        /*tvecs = */ tvecs,
        /*perViewErrors = */ cv::noArray(),
        /*flags = */ cv::CALIB_FIX_INTRINSIC,
        /*termCriteria = */ cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 8, 1e-3) // defaults are 30, 1e-6 -- we only need a rough estimate
      );

      printf("cv::stereoCalibrate() camera %zu to %zu\n", leftCameraIdx, rightCameraIdx);
      printf("  %zu observations, %.3f ms\n", objectPoints.size(), perfTimer.checkpoint());

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
      if (leftCameraIdx == 0) {
        base_to_camera_guess[rightCameraIdx] = stereoOffset;
      }
    }));
  }

  // Ensure all array tasks have completed.
  for (const auto& taskCompletion : taskCompletions) {
    taskCompletion();
  }


#endif // OpenCV stereo calibration

  // Multi-camera
  printf("=== industrial_calibration multi-camera calibration ===\n");
  {
    industrial_calibration::ExtrinsicMultiStaticCameraOnlyProblem problem;

    problem.fix_first_camera = true;

    // Per-camera intrinsics
    for (size_t cameraIdx = 0; cameraIdx < data.cameras.size(); ++cameraIdx) {
      CameraCalibrationData& cameraData = data.cameras[cameraIdx];
      problem.intr.push_back(cameraData.intrinsicCalibration.intrinsics);
    }

    // Sanity check on observation counts
    for (size_t cameraIdx = 1; cameraIdx < data.cameras.size(); ++cameraIdx) {
      assert(data.cameras[cameraIdx].observations.size() == data.cameras[0].observations.size());
    }

    // Per-camera base-to-camera guesses -- use cv::stereoCalibrate guesses
#if 0
    problem.base_to_camera_guess = base_to_camera_guess;
#else
    problem.base_to_camera_guess = std::vector(data.cameras.size(), Eigen::Isometry3d::Identity());
#endif

    // Per-observation base-to-target guesses.
    // We initialize this with the target transforms from the intrinsic calibration process.
    // Not all target transforms will be available for the base camera (camera 0), though, so we use other camera
    // and compose their initial base-to-camera guesses to come up with base-to-target guesses.

    problem.base_to_target_guess.resize(data.observationCount());
    for (size_t i = 0; i < problem.base_to_target_guess.size(); ++i) {
      problem.base_to_target_guess[i] = data.cameras[0].observations[i].targetTransform;

      bool isValid = !data.cameras[0].observations[i].empty();
      size_t pointCount = data.cameras[0].observations[i].pointCount();

      // printIsometry("Base-to-target guess from camera 0: ", problem.base_to_target_guess[i], "\n");

      for (size_t cameraIdx = 1; cameraIdx < data.cameras.size(); ++cameraIdx) {
        if (data.cameras[cameraIdx].observations[i].targetTransform.isApprox(Eigen::Isometry3d::Identity()))
          continue;

        Eigen::Isometry3d xf = base_to_camera_guess[cameraIdx] * data.cameras[cameraIdx].observations[i].targetTransform;

        // printf("  Base-to-target guess transformed from camera %zu: ", cameraIdx);
        // printIsometry(xf, "\n");

        if (!isValid) {
          // For observations that don't have a valid base transform, compose a guess transform from the other
          // camera that saw the most points.
          if (data.cameras[cameraIdx].observations[i].pointCount() > pointCount) {
            pointCount = data.cameras[cameraIdx].observations[i].pointCount();
            problem.base_to_target_guess[i] = xf;
          }
        }
      }
    }

    // Set up image observations
    problem.image_observations.resize(data.cameras.size());
    for (size_t cameraIdx = 0; cameraIdx < data.cameras.size(); ++cameraIdx) {
      CameraCalibrationData& cameraData = data.cameras[cameraIdx];
      std::vector<industrial_calibration::Correspondence2D3D::Set>& cameraObsSets = problem.image_observations[cameraIdx];
      cameraObsSets.resize(cameraData.observations.size());

      for (size_t obsIdx = 0; obsIdx < cameraData.observations.size(); ++obsIdx) {
        cameraObsSets[obsIdx] = cameraData.observations[obsIdx].correspondenceSet();

      }
    }

    // Run solver
    perfTimer.checkpoint();
    industrial_calibration::ExtrinsicMultiStaticCameraOnlyResult result = optimize(problem);

    printf("Optimization %s in %.3f ms\n", (result.converged ? "converged" : "did not converge"), perfTimer.checkpoint());

    printf("Initial cost per observation (pixels): %.3f\n", std::sqrt(result.initial_cost_per_obs));
    printf("Final cost per observation (pixels): %.3f\n", std::sqrt(result.final_cost_per_obs));
    printf("\n");


    if (result.converged) {
#if 0
      for (size_t i = 0; i < result.base_to_target.size(); ++i) {
        printf("Base-to-target %zu: ", i);
        printIsometry(result.base_to_target[i], "\n");
      }
#endif
      for (size_t i = 0; i < result.base_to_camera.size(); ++i) {
        printf("Base-to-camera %zu: ", i);
        printIsometry(result.base_to_camera[i], "\n");
      }

      // Composed transforms

      for (size_t srcIdx = 0; srcIdx < data.cameras.size(); ++srcIdx) {
        for (size_t dstIdx = 0; dstIdx < data.cameras.size(); ++dstIdx) {
          Eigen::Isometry3d xf = result.base_to_camera[srcIdx].inverse() * result.base_to_camera[dstIdx];

          printf("Camera %zu to camera %zu: ", srcIdx, dstIdx);
          printIsometry(xf, "\n");
        }
      }
    }



    // Save calibration data
    cv::FileStorage fs("calibration.yml", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);

    fs.startWriteStruct("cameras", cv::FileNode::SEQ, cv::String());
    for (size_t cameraIdx = 0; cameraIdx < data.cameras.size(); ++cameraIdx) {
      fs.startWriteStruct(cv::String(), cv::FileNode::MAP, cv::String());
      CameraCalibrationData& cameraData = data.cameras[cameraIdx];

      fs.write("intrinsicMatrix", cameraData.cvIntrinsicMatrix());
      fs.write("distortionCoeffs", cameraData.cvDistCoeffs());

      fs.endWriteStruct();
    }
    fs.endWriteStruct(); // cameras

    fs.startWriteStruct("views", cv::FileNode::SEQ, cv::String());

    // View indices (L, R): [2, 0] [1, 3]
    // View stereo offset should be along -X axis
    data.views.resize(2);
    data.views[0].cameraIndices[0] = 2;
    data.views[0].cameraIndices[1] = 0;
    data.views[1].cameraIndices[0] = 1;
    data.views[1].cameraIndices[1] = 3;


    // View offset calculation attempt via point triangulation
    // Find common points across all views

    std::vector<cv::Point2f> commonImagePoints[4];

    for (size_t observationIdx = 0; observationIdx < data.observationCount(); ++observationIdx) {
      // Intersection of all point ID sets
      std::set<int> commonPointIds;

      // Start with all points in camera 0 view
      commonPointIds.insert(data.cameras[0].observations[observationIdx].objectPointIds.begin(), data.cameras[0].observations[observationIdx].objectPointIds.end());
      if (commonPointIds.empty())
        continue; // fast-path exit

      for (size_t cameraIdx = 1; cameraIdx < 4; ++cameraIdx) {
        const auto& obs = data.cameras[cameraIdx].observations[observationIdx];
        std::set<int> newCommonPointIds;
        std::set_intersection(commonPointIds.begin(), commonPointIds.end(), obs.objectPointIds.begin(), obs.objectPointIds.end(), std::inserter(newCommonPointIds, std::end(newCommonPointIds)));
        newCommonPointIds.swap(commonPointIds);
      }

      if (commonPointIds.empty())
        continue;

      // Collect points from all views.
      // (This assumes that the points are sorted by ID inside the view data)
      for (size_t cameraIdx = 0; cameraIdx < 4; ++cameraIdx) {
        const auto& cameraObs = data.cameras[cameraIdx].observations[observationIdx];
        for (size_t pointIdx = 0; pointIdx < cameraObs.objectPointIds.size(); ++pointIdx) {
          if (commonPointIds.find(cameraObs.objectPointIds[pointIdx]) != commonPointIds.end()) {

            commonImagePoints[cameraIdx].push_back(cameraObs.imagePoints[pointIdx]);
          }
        }
      }

      // Sanity check -- all point vectors should be the same size.
      for (size_t cameraIdx = 1; cameraIdx < 4; ++cameraIdx) {
        assert(commonImagePoints[cameraIdx].size() == commonImagePoints[0].size());
      }
    }

    printf("%zu common points across all 4 cameras\n", commonImagePoints[0].size());




    for (size_t viewIdx = 0; viewIdx < data.views.size(); ++viewIdx) {
      // Compute view stereoRotation and stereoTranslation from the base_to_camera info
      ViewCalibrationData& viewData = data.views[viewIdx];

      int leftCameraIdx = viewData.cameraIndices[0];
      int rightCameraIdx = viewData.cameraIndices[1];

      CameraCalibrationData& leftC = data.cameras[leftCameraIdx];
      CameraCalibrationData& rightC = data.cameras[rightCameraIdx];

      // For the stereo translation and rotation, we need a composed transform from the right camera to the left camera.
      Eigen::Isometry3d xf = result.base_to_camera[rightCameraIdx].inverse() * result.base_to_camera[leftCameraIdx];

      Eigen::Vector3d tx = xf.translation();
      Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rx = xf.rotation(); // RowMajor order for interop with OpenCV

      viewData.stereoTranslation = cv::Mat(/*rows=*/ 3, /*cols=*/ 1, CV_64F, tx.data()).clone();
      viewData.stereoRotation = cv::Mat(/*rows=*/ 3, /*cols=*/ 3, CV_64F, rx.data()).clone();


      // Stereo rectify the 2 cameras
      cv::stereoRectify(
        leftC.cvIntrinsicMatrix(), leftC.cvDistCoeffs(),
        rightC.cvIntrinsicMatrix(), rightC.cvDistCoeffs(),
        leftC.imageSize,
        viewData.stereoRotation, viewData.stereoTranslation,
        viewData.stereoRectification[0], viewData.stereoRectification[1],
        viewData.stereoProjection[0], viewData.stereoProjection[1],
        viewData.stereoDisparityToDepth,
        /*flags=*/cv::CALIB_ZERO_DISPARITY, /*alpha=*/ -1.0f, cv::Size(),
        &viewData.stereoValidROI[0], &viewData.stereoValidROI[1]);

      printf("View %zu\n", viewIdx);

      for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
        printf("\n ===== View %zu | %s Camera (%u) ===== \n", viewIdx, eyeIdx == 0 ? "Left" : "Right", viewData.cameraIndices[eyeIdx]);

        std::cout << "* Stereo Rectification matrix:" << std::endl << viewData.stereoRectification[eyeIdx] << std::endl;
        std::cout << "* Stereo Projection matrix:" << std::endl << viewData.stereoProjection[eyeIdx] << std::endl;
        std::cout << "* Stereo Valid ROI:" << std::endl << viewData.stereoValidROI[eyeIdx] << std::endl;
      }
    }

    // Undistort points following the same convention used in CameraSystem (same parameters passed to cv::initUndistortRectifyMap)


    // View index -> vector<Point3f>
    std::vector<std::vector<cv::Point3f>> triangulatedPoints;
    triangulatedPoints.resize(data.views.size());

    for (size_t viewIdx = 0; viewIdx < data.views.size(); ++viewIdx) {
      ViewCalibrationData& viewData = data.views[viewIdx];

      cv::Mat viewUndistortedPoints[2];

      for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
        int cameraIdx = viewData.cameraIndices[eyeIdx];
        CameraCalibrationData& c = data.cameras[cameraIdx];

        cv::undistortPoints(
          /*src=*/ commonImagePoints[cameraIdx],
          /*dst=*/ viewUndistortedPoints[eyeIdx],
          /*cameraMatrix=*/ c.cvIntrinsicMatrix(),
          /*distCoeffs=*/ c.cvDistCoeffs(),
          /*R=*/ viewData.stereoRectification[eyeIdx],
          /*P=*/ viewData.stereoProjection[eyeIdx]);
      }

      // Triangulate points
      cv::Mat hPoints; // homogenous output points
      cv::triangulatePoints(
        /*projMatr1=*/ viewData.stereoProjection[0],
        /*projMatr2=*/ viewData.stereoProjection[1],
        /*projPoints1=*/ viewUndistortedPoints[0],
        /*projPoints2=*/ viewUndistortedPoints[1],
        hPoints);

      assert(hPoints.rows == 4 && hPoints.type() == CV_32F);
      for (size_t col = 0; col < hPoints.cols; ++col) {
        // Convert from homogenous coordinates and apply coordinate system convention flip
        float x =  hPoints.at<float>(/*row=*/ 0, col);
        float y = -hPoints.at<float>(/*row=*/ 1, col);
        float z = -hPoints.at<float>(/*row=*/ 2, col);
        float w =  hPoints.at<float>(/*row=*/ 3, col);
        triangulatedPoints[viewIdx].push_back(cv::Point3f(x / w, y / w, z / w));
      }

      printf("View %zu: triangulated %zu points\n", viewIdx, triangulatedPoints[viewIdx].size());
    }

    for (size_t viewIdx = 1; viewIdx < data.views.size(); ++viewIdx) {
      // Find linear transform between this view's points and view 0
      cv::Mat transform = cv::estimateAffine3D(triangulatedPoints[viewIdx], triangulatedPoints[0]);
      assert(transform.type() == CV_64F);
      assert(transform.rows == 3 && transform.cols == 4);
      std::cout << transform << std::endl;

      // Convert transform to an Isometry
      Eigen::Isometry3d pose;
      pose.translation() = Eigen::Vector3d(transform.at<double>(0, 3), transform.at<double>(1, 3), transform.at<double>(2, 3));
      pose.linear() = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Map(
        transform(cv::Range(0, 3), cv::Range(0, 3)).clone().ptr<double>());

      printIsometry(pose, "\n");
    }


    for (size_t viewIdx = 0; viewIdx < data.views.size(); ++viewIdx) {
      ViewCalibrationData& viewData = data.views[viewIdx];
      int leftCameraIdx = viewData.cameraIndices[0];
      int rightCameraIdx = viewData.cameraIndices[1];

      fs.startWriteStruct(cv::String(), cv::FileNode::MAP, cv::String());
      fs.write("isStereo", 1);
      fs.write("isPanorama", 0);
      fs.write("leftCameraIndex", (int) leftCameraIdx);
      fs.write("rightCameraIndex", (int) rightCameraIdx);
      fs.write("stereoTranslation", viewData.stereoTranslation);
      fs.write("stereoRotation", viewData.stereoRotation);


      // Compose a transform for the view offset, if this is not the first view.
      if (viewIdx == 0) {
        fs.write("viewTranslation", cv::Mat::zeros(/*rows=*/ 3, /*cols=*/ 1, CV_64F));
        fs.write("viewRotation", cv::Mat::zeros(/*rows=*/ 3, /*cols=*/ 1, CV_64F));
      } else {

        // Transform from the view 0 left camera to the current view's left camera
        Eigen::Isometry3d xf = result.base_to_camera[data.views[0].cameraIndices[0]].inverse() * result.base_to_camera[leftCameraIdx];

        Eigen::Vector3d tx = xf.translation();
        Eigen::Vector3d rxEulerDeg = Eigen::EulerAnglesYXZd(xf.rotation()).angles() * (180.0 / M_PI);

        fs.write("viewTranslation", cv::Mat(/*rows=*/ 3, /*cols=*/ 1, CV_64F, tx.data()));
        fs.write("viewRotation", cv::Mat(/*rows=*/ 3, /*cols=*/ 1, CV_64F, rxEulerDeg.data()));
      }

      fs.endWriteStruct();
    }

    fs.endWriteStruct(); // views
    
  }



  FxThreading::detail::shutdown();

  return 0;
}

