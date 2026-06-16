#pragma once
#include "RollingShutterTiming.h"
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace CameraImuCalib {

// Fixed camera intrinsics + distortion for one camera, read from calibration.yml. Treated
// as fixed throughout -- never optimized. Wraps OpenCV's 5-coefficient model so
// Stage 1 (undistort) and the synthetic generator (project) share one distortion path.
class Intrinsics {
public:
  // Load the camera at `cameraIndex` from the `cameras` sequence of calibration.yml. Also
  // pulls readoutBottomToTop into `outTiming.flipReadout`. Returns false on failure.
  bool load(const std::string& calibrationYamlPath, int cameraIndex, RollingShutterTiming& outTiming);

  // Build directly from parameters (used by the synthetic self-test).
  void set(double fx, double fy, double cx, double cy, const std::vector<double>& distCoeffs,
    const cv::Size& imageSize);

  const cv::Matx33d& cameraMatrix() const { return m_cameraMatrix; }
  const std::vector<double>& distCoeffs() const { return m_distCoeffs; }
  cv::Size imageSize() const { return m_imageSize; }

  double fx() const { return m_cameraMatrix(0, 0); }
  double fy() const { return m_cameraMatrix(1, 1); }
  double cx() const { return m_cameraMatrix(0, 2); }
  double cy() const { return m_cameraMatrix(1, 2); }

  // Undistort pixel points to normalized camera coordinates x = (u-cx)/fx-equivalent with
  // distortion removed (cv::undistortPoints with no P). Batched for one frame pair.
  void undistortToNormalized(const std::vector<cv::Point2d>& pixels,
    std::vector<cv::Point2d>& outNormalized) const;

  // Project normalized camera-frame rays (x, y, 1) to distorted pixels (cv::projectPoints
  // with the distortion model applied). Inverse of undistortToNormalized; used to synthesize
  // flow endpoints.
  void projectFromNormalized(const std::vector<cv::Point2d>& normalized,
    std::vector<cv::Point2d>& outPixels) const;

private:
  cv::Matx33d m_cameraMatrix = cv::Matx33d::eye();
  std::vector<double> m_distCoeffs = {0, 0, 0, 0, 0};
  cv::Size m_imageSize = cv::Size(1920, 1080);
};

} // namespace CameraImuCalib
