#include "SyntheticFlowSource.h"
#include "Conventions.h"
#include <algorithm>
#include <cmath>

namespace CameraImuCalib {

bool SyntheticFlowSource::pair(size_t pairIndex, FramePairFlow& outPair) {
  if (pairIndex >= m_pairs.size())
    return false;
  const PairSpec& spec = m_pairs[pairIndex];
  outPair.frameStartTimeA = spec.frameStartTimeA;
  outPair.frameStartTimeB = spec.frameStartTimeB;
  outPair.samples.clear();

  // Build the strided pixel grid for frame i.
  std::vector<cv::Point2d> px0;
  for (int v = m_stride / 2; v < m_height; v += m_stride)
    for (int u = m_stride / 2; u < m_width; u += m_stride)
      px0.emplace_back(static_cast<double>(u), static_cast<double>(v));

  // Undistort -> normalized rays, rotate by dRCam^-1, reproject to distorted pixels.
  std::vector<cv::Point2d> norm0;
  m_intrinsics.undistortToNormalized(px0, norm0);

  if (m_mode == Mode::kRowTimedLinear) {
    // Validator: flow_i = M(x,y) * (omega0 + alpha*tau_i) * dt_i, with per-vector row
    // timing. y2 (end row) depends on the flow, so iterate a short fixed point.
    const double dtCenter = spec.frameStartTimeB - spec.frameStartTimeA;
    const Eigen::Vector3d omega0 = logSO3(spec.dRCam) / dtCenter;
    const Eigen::Vector3d alpha = spec.alphaCam;
    const double tMidPair = 0.5 * (m_timing.frameCenterTime(spec.frameStartTimeA) + m_timing.frameCenterTime(spec.frameStartTimeB));
    auto clampRow = [&](double y) {
      return static_cast<uint32_t>(std::min(std::max(0, static_cast<int>(std::lround(y))), m_height - 1));
    };
    std::vector<cv::Point2d> norm1(norm0.size()), px1;
    std::vector<double> y2(norm0.size());
    for (size_t i = 0; i < norm0.size(); ++i)
      y2[i] = px0[i].y; // init end row = start row
    for (int iter = 0; iter < 2; ++iter) {
      for (size_t i = 0; i < norm0.size(); ++i) {
        const double x = norm0[i].x, y = norm0[i].y;
        const double ta = m_timing.rowExposureTime(spec.frameStartTimeA, clampRow(px0[i].y));
        const double tb = m_timing.rowExposureTime(spec.frameStartTimeB, clampRow(y2[i]));
        const double dt = tb - ta;
        const double tau = 0.5 * (ta + tb) - tMidPair;
        const Eigen::Vector3d omega = omega0 + alpha * tau;
        const double xdot = (x * y * omega.x() - (1.0 + x * x) * omega.y() + y * omega.z()) * dt;
        const double ydot = ((1.0 + y * y) * omega.x() - x * y * omega.y() - x * omega.z()) * dt;
        norm1[i] = cv::Point2d(x + xdot, y + ydot);
      }
      m_intrinsics.projectFromNormalized(norm1, px1);
      for (size_t i = 0; i < px1.size(); ++i)
        y2[i] = px1[i].y;
    }
    outPair.samples.reserve(px0.size());
    for (size_t i = 0; i < px0.size(); ++i) {
      FlowSample s;
      s.pixel0 = Eigen::Vector2d(px0[i].x, px0[i].y);
      s.pixel1 = Eigen::Vector2d(px1[i].x, px1[i].y);
      s.confidence = 1.0;
      outPair.samples.push_back(s);
    }
    return true;
  }

  std::vector<cv::Point2d> norm1;
  norm1.reserve(norm0.size());
  if (m_mode == Mode::kProjection) {
    const Eigen::Quaterniond invRotation = spec.dRCam.conjugate();
    for (const cv::Point2d& n : norm0) {
      const Eigen::Vector3d d0(n.x, n.y, 1.0);
      const Eigen::Vector3d d1 = invRotation * d0;
      norm1.emplace_back(d1.x() / d1.z(), d1.y() / d1.z());
    }
  } else {
    // Linear flow of theta = log(dRCam): xdot/ydot added to the normalized point.
    const Eigen::Vector3d theta = logSO3(spec.dRCam);
    for (const cv::Point2d& n : norm0) {
      const double x = n.x, y = n.y;
      const double xdot = x * y * theta.x() - (1.0 + x * x) * theta.y() + y * theta.z();
      const double ydot = (1.0 + y * y) * theta.x() - x * y * theta.y() - x * theta.z();
      norm1.emplace_back(x + xdot, y + ydot);
    }
  }

  std::vector<cv::Point2d> px1;
  m_intrinsics.projectFromNormalized(norm1, px1);

  outPair.samples.reserve(px0.size());
  for (size_t i = 0; i < px0.size(); ++i) {
    FlowSample s;
    s.pixel0 = Eigen::Vector2d(px0[i].x, px0[i].y);
    s.pixel1 = Eigen::Vector2d(px1[i].x, px1[i].y);
    s.confidence = 1.0;
    outPair.samples.push_back(s);
  }
  return true;
}

} // namespace CameraImuCalib
