#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <type_traits>

// =====================================================================================
// Conventions for the camera-IMU autosync calibrator.
//
// This header is the single source of truth for the frame, sign, and unit conventions.
// Everything downstream (estimator stages and the Ceres refinement residual) depends on
// these definitions, so they live in exactly one place.
//
// All rotation helpers are templated on the scalar type so the identical code runs both
// on plain `double` (initialization, diagnostics) and on `ceres::Jet` (autodiff through
// the refinement residual). The small-angle Taylor branches exist precisely so the
// functions stay finite and smoothly differentiable as the rotation angle approaches 0.
//
// Frames:
//   Camera frame: x right, y down, z forward (optical axis). Normalized image coords
//     x = (u - cx)/fx, y = (v - cy)/fy after undistortion.
//   IMU/body frame: gyro reports body angular rate in deg/s.
//
//   R_imu_to_cam rotates an IMU-frame vector into the camera frame:
//     v_cam = R_imu_to_cam * v_imu
//
//   Angular-velocity transform:  omega_cam = R_imu_to_cam * (omega_imu - b)
//   Relative-rotation transform: dR_cam   = R_imu_to_cam * dR_imu * R_imu_to_cam^-1
//
// Time offset sign: t_imu = t_cam + delta.  Adding `delta` to a camera time yields
//   the corresponding instant on the IMU clock.
// =====================================================================================

namespace CameraImuCalib {

// Documented relation between the camera and IMU clocks. Restated in the output header verbatim.
inline constexpr const char* kTimeOffsetRelation = "t_imu = t_cam + delta";

// All angles in human-facing reports are degrees. Scalar forms are constrained to
// arithmetic types so the Eigen-vector overloads below are unambiguous.
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
inline T radToDeg(const T& r) { return r * T(180.0 / M_PI); }
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
inline T degToRad(const T& d) { return d * T(M_PI / 180.0); }

// Vector forms (deg/s gyro vectors, etc.); element type may be double or ceres::Jet.
template <typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, Derived::RowsAtCompileTime, Derived::ColsAtCompileTime>
degToRad(const Eigen::MatrixBase<Derived>& v) {
  using Scalar = typename Derived::Scalar;
  return v * Scalar(M_PI / 180.0);
}

// Exponential map: rotation vector (axis * angle, radians) -> unit quaternion.
// Right-handed, exp([omega]x). Composing these in time order integrates a rotation.
template <typename T>
Eigen::Quaternion<T> expSO3(const Eigen::Matrix<T, 3, 1>& omega) {
  // ADL idiom: for ceres::Jet these resolve to ceres::{sin,cos,sqrt}; for double, std::.
  using std::cos;
  using std::sin;
  using std::sqrt;
  const T thetaSq = omega.squaredNorm();
  const T theta = sqrt(thetaSq);

  // w = cos(theta/2); xyz = (sin(theta/2)/theta) * omega.
  // The vector scale sin(theta/2)/theta -> 1/2 - theta^2/48 as theta -> 0; use the
  // Taylor form below the threshold so the expression stays finite and differentiable.
  T halfAngle = theta * T(0.5);
  T vectorScale;
  if (theta > T(1e-8)) {
    vectorScale = sin(halfAngle) / theta;
  } else {
    // sin(theta/2)/theta = 1/2 - theta^2/48 + ...
    vectorScale = T(0.5) - thetaSq * T(1.0 / 48.0);
  }

  Eigen::Quaternion<T> q;
  q.w() = cos(halfAngle);
  q.x() = omega.x() * vectorScale;
  q.y() = omega.y() * vectorScale;
  q.z() = omega.z() * vectorScale;
  return q;
}

// Logarithm map: unit quaternion -> rotation vector (axis * angle, radians).
// Inverse of expSO3 on the principal branch (angle in [0, pi]).
template <typename T>
Eigen::Matrix<T, 3, 1> logSO3(const Eigen::Quaternion<T>& qIn) {
  using std::atan2;
  using std::sqrt;
  Eigen::Quaternion<T> q = qIn;
  // Canonicalize to the hemisphere with w >= 0 so we recover the minimal-angle vector.
  if (q.w() < T(0.0)) {
    q.w() = -q.w();
    q.x() = -q.x();
    q.y() = -q.y();
    q.z() = -q.z();
  }

  const Eigen::Matrix<T, 3, 1> v(q.x(), q.y(), q.z());
  const T vNormSq = v.squaredNorm();
  const T vNorm = sqrt(vNormSq);

  // angle = 2 * atan2(|v|, w); rotation vector = v * (angle / |v|).
  // As |v| -> 0 the factor angle/|v| -> 2/w (a smooth limit); use the Taylor form
  // below the threshold to keep the division well-behaved under autodiff.
  T scale;
  if (vNorm > T(1e-8)) {
    scale = T(2.0) * atan2(vNorm, q.w()) / vNorm;
  } else {
    // 2*atan2(|v|,w)/|v| = (2/w) * (1 - |v|^2/(3 w^2) + ...); w ~= 1 near identity.
    scale = T(2.0) / q.w() * (T(1.0) - vNormSq / (T(3.0) * q.w() * q.w()));
  }
  return v * scale;
}

// Apply the relative-rotation transform: dR_cam = R * dR_imu * R^-1, expressed on
// quaternions. `qImuToCam` is R_imu_to_cam; `dRImu` is the body-frame relative rotation.
template <typename T>
Eigen::Quaternion<T> transformRelativeRotation(const Eigen::Quaternion<T>& qImuToCam,
  const Eigen::Quaternion<T>& dRImu) {
  return qImuToCam * dRImu * qImuToCam.conjugate();
}

} // namespace CameraImuCalib
