#pragma once
#include "Conventions.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/jet.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace CameraImuCalib {

// Extract the scalar value of a templated scalar so we can do (non-differentiable) index
// selection against the double-valued sample timestamps. The moving integration boundary
// stays differentiable because the interpolated endpoint sample's dt is computed in T.
inline double scalarValue(double v) { return v; }
template <typename T, int N>
inline double scalarValue(const ceres::Jet<T, N>& v) {
  return scalarValue(v.a);
}

// One IMU sample on the recorded (zero-based) capture clock.
struct ImuSample {
  double time = 0.0; // seconds, on the t_imu clock (CalibrationWriter timebase)
  Eigen::Vector3d gyroDeg; // body angular rate, deg/s, NOT bias-compensated
  Eigen::Vector3d accelG; // specific force, G
};

// Recorded IMU trace (CalibrationWriter imu.csv): one row per sample at ~3600 Hz,
//   timestamp(ns), gyro_x, gyro_y, gyro_z (deg/s), accel_x, accel_y, accel_z (G)
// with a leading header line that is skipped. Trailing fields, if any, are ignored.
class ImuTrace {
public:
  // Load from CSV. Skips the header line; rejects duplicate or backward timestamps with a
  // logged warning (keeping the monotonic prefix). Returns false on open/parse failure.
  bool load(const std::string& path);

  size_t sampleCount() const { return m_samples.size(); }
  bool isEmpty() const { return m_samples.empty(); }
  double firstTime() const { return m_samples.front().time; }
  double lastTime() const { return m_samples.back().time; }
  const ImuSample& sample(size_t i) const { return m_samples[i]; }

  // True if [t0, t1] lies fully within the recorded sample span (so integration over the
  // window never has to clamp to an endpoint sample).
  bool covers(double t0, double t1) const {
    return !m_samples.empty() && t0 >= firstTime() && t1 <= lastTime();
  }

  // Linearly-interpolated gyro (deg/s) at time t. Clamps to the endpoint sample outside
  // the recorded span. Templated so it differentiates w.r.t. t (used for moving window
  // edges driven by delta).
  template <typename T>
  Eigen::Matrix<T, 3, 1> gyroDegAt(const T& t) const {
    if (m_samples.empty())
      return Eigen::Matrix<T, 3, 1>::Zero();

    const double ts = scalarValue(t);
    if (ts <= m_times.front())
      return m_samples.front().gyroDeg.cast<T>();
    if (ts >= m_times.back())
      return m_samples.back().gyroDeg.cast<T>();

    // First sample strictly after ts; interpolate within [lo, hi].
    const size_t hi = static_cast<size_t>(
      std::upper_bound(m_times.begin(), m_times.end(), ts) - m_times.begin());
    const size_t lo = hi - 1;
    const T frac = (t - T(m_times[lo])) / T(m_times[hi] - m_times[lo]);
    return m_samples[lo].gyroDeg.cast<T>() +
      frac * (m_samples[hi].gyroDeg - m_samples[lo].gyroDeg).cast<T>();
  }

  // Integrate the body-frame relative rotation dR_imu over [t0, t1] from the gyro, with a
  // constant bias subtracted: omega = gyro - biasDeg. Trapezoidal in the rotation vector
  // and composed in time order via the exponential map. Fractional endpoints use
  // gyroDegAt() so the residual is smooth and differentiable in delta (which shifts t0,t1)
  // and in biasDeg. At 3600 Hz a typical window holds ~tens of samples.
  template <typename T>
  Eigen::Quaternion<T> integrateBodyRotation(const T& t0, const T& t1,
    const Eigen::Matrix<T, 3, 1>& biasDeg) const {
    Eigen::Quaternion<T> q = Eigen::Quaternion<T>::Identity();
    const double s0 = scalarValue(t0);
    const double s1 = scalarValue(t1);
    if (m_samples.empty() || s1 <= s0)
      return q;

    // Walk knot times: t0, every sample time strictly inside (t0, t1), then t1.
    size_t i = static_cast<size_t>(
      std::upper_bound(m_times.begin(), m_times.end(), s0) - m_times.begin());

    T prevTime = t0;
    Eigen::Matrix<T, 3, 1> prevGyro = gyroDegAt(t0);
    for (; i < m_times.size() && m_times[i] < s1; ++i) {
      const T curTime = T(m_times[i]);
      const Eigen::Matrix<T, 3, 1> curGyro = m_samples[i].gyroDeg.cast<T>();
      accumulateSegment(q, prevTime, curTime, prevGyro, curGyro, biasDeg);
      prevTime = curTime;
      prevGyro = curGyro;
    }
    const Eigen::Matrix<T, 3, 1> endGyro = gyroDegAt(t1);
    accumulateSegment(q, prevTime, t1, prevGyro, endGyro, biasDeg);

    q.normalize();
    return q;
  }

  // Static-segment bias init: scan for the longest contiguous run with |gyro| below
  // maxRateDegPerSec lasting at least minDurationSec, and return its mean gyro as the bias.
  // Returns false if no qualifying run exists (caller falls back to zero bias).
  bool findStaticBias(double maxRateDegPerSec, double minDurationSec,
    Eigen::Vector3d& outBiasDeg) const;

private:
  template <typename T>
  void accumulateSegment(Eigen::Quaternion<T>& q, const T& ta, const T& tb,
    const Eigen::Matrix<T, 3, 1>& gyroA,
    const Eigen::Matrix<T, 3, 1>& gyroB,
    const Eigen::Matrix<T, 3, 1>& biasDeg) const {
    const T dt = tb - ta;
    // Trapezoidal mean rate over the sub-interval, bias-corrected, deg/s -> rad/s.
    const Eigen::Matrix<T, 3, 1> meanGyroDeg = T(0.5) * (gyroA + gyroB) - biasDeg;
    const Eigen::Matrix<T, 3, 1> dTheta = degToRad(meanGyroDeg) * dt;
    q = q * expSO3(dTheta); // compose body-frame increment in time order
  }

  std::vector<ImuSample> m_samples;
  std::vector<double> m_times; // parallel to m_samples, for binary search
};

} // namespace CameraImuCalib
