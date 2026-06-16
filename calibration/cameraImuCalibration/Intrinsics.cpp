#include "Intrinsics.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/persistence.hpp>
#include <cstdio>

namespace CameraImuCalib {

bool Intrinsics::load(const std::string& calibrationYamlPath, int cameraIndex,
  RollingShutterTiming& outTiming) {
  cv::FileStorage fs(calibrationYamlPath, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    fprintf(stderr, "Intrinsics::load: cannot open '%s'\n", calibrationYamlPath.c_str());
    return false;
  }

  cv::FileNode cameras = fs["cameras"];
  if (cameras.empty() || !cameras.isSeq()) {
    fprintf(stderr, "Intrinsics::load: '%s' has no 'cameras' sequence\n", calibrationYamlPath.c_str());
    return false;
  }
  if (cameraIndex < 0 || cameraIndex >= static_cast<int>(cameras.size())) {
    fprintf(stderr, "Intrinsics::load: camera index %d out of range (have %d cameras)\n",
      cameraIndex, static_cast<int>(cameras.size()));
    return false;
  }

  cv::FileNode cam = cameras[cameraIndex];
  cv::Mat intrinsicMatrix, distortion;
  cam["intrinsicMatrix"] >> intrinsicMatrix;
  cam["distortionCoeffs"] >> distortion;
  if (intrinsicMatrix.rows != 3 || intrinsicMatrix.cols != 3) {
    fprintf(stderr, "Intrinsics::load: camera %d intrinsicMatrix is not 3x3\n", cameraIndex);
    return false;
  }

  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      m_cameraMatrix(r, c) = intrinsicMatrix.at<double>(r, c);

  m_distCoeffs.assign(distortion.begin<double>(), distortion.end<double>());
  if (m_distCoeffs.empty())
    m_distCoeffs = {0, 0, 0, 0, 0};

  // readoutBottomToTop is optional; default top-to-bottom (false).
  int readoutBottomToTop = 0;
  if (!cam["readoutBottomToTop"].empty())
    cam["readoutBottomToTop"] >> readoutBottomToTop;
  outTiming.flipReadout = (readoutBottomToTop != 0);

  // calibration.yml does not carry image size; default to the capture resolution. The
  // estimator overrides visibleHeight from the actual frames if needed.
  m_imageSize = cv::Size(1920, 1080);

  printf("Intrinsics: camera %d  fx=%.2f fy=%.2f cx=%.2f cy=%.2f  flipReadout=%d  dist=[",
    cameraIndex, fx(), fy(), cx(), cy(), outTiming.flipReadout ? 1 : 0);
  for (size_t i = 0; i < m_distCoeffs.size(); ++i)
    printf("%s%.5g", i ? ", " : "", m_distCoeffs[i]);
  printf("]\n");
  return true;
}

void Intrinsics::set(double fx, double fy, double cx, double cy,
  const std::vector<double>& distCoeffs, const cv::Size& imageSize) {
  m_cameraMatrix = cv::Matx33d::eye();
  m_cameraMatrix(0, 0) = fx;
  m_cameraMatrix(1, 1) = fy;
  m_cameraMatrix(0, 2) = cx;
  m_cameraMatrix(1, 2) = cy;
  m_distCoeffs = distCoeffs.empty() ? std::vector<double>{0, 0, 0, 0, 0} : distCoeffs;
  m_imageSize = imageSize;
}

void Intrinsics::undistortToNormalized(const std::vector<cv::Point2d>& pixels,
  std::vector<cv::Point2d>& outNormalized) const {
  outNormalized.clear();
  if (pixels.empty())
    return;
  // No P matrix -> output is in normalized camera coordinates. Iterative undistortion is
  // OpenCV's default and is accurate for these small distortions.
  cv::undistortPoints(pixels, outNormalized, cv::Mat(m_cameraMatrix), m_distCoeffs);
}

void Intrinsics::projectFromNormalized(const std::vector<cv::Point2d>& normalized,
  std::vector<cv::Point2d>& outPixels) const {
  outPixels.clear();
  if (normalized.empty())
    return;
  std::vector<cv::Point3d> rays;
  rays.reserve(normalized.size());
  for (const cv::Point2d& n : normalized)
    rays.emplace_back(n.x, n.y, 1.0);

  const cv::Vec3d zero(0, 0, 0);
  cv::projectPoints(rays, zero, zero, cv::Mat(m_cameraMatrix), m_distCoeffs, outPixels);
}

} // namespace CameraImuCalib
