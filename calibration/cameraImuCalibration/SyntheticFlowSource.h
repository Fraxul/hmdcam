#pragma once
#include "FlowSource.h"
#include "Intrinsics.h"
#include "RollingShutterTiming.h"
#include <Eigen/Geometry>
#include <vector>

namespace CameraImuCalib {

// Generates dense flow for the synthetic round-trip self-test. For each frame pair it
// projects a known camera-frame relative rotation through the intrinsics (distortion applied
// on the way out), so the estimator's full undistort + fit path is exercised end to end.
//
// A world-fixed ray d (frame i) re-expressed in the rotated camera frame is d' = dRCam^-1 d
// (the conjugate). With that sign, Stage 1's linear fit recovers theta_cam = log(dRCam),
// which is exactly what the refinement predicts as R * dR_imu * R^-1. (Translation/parallax
// is intentionally absent: this is the rotation-only model.)
class SyntheticFlowSource : public FlowSource {
public:
  // How flow endpoints are generated:
  //  kProjection  -- rotate the ray by dRCam^-1 and reproject (exact; exercises the full
  //                  undistort path, but Stage 1's linear fit sees an O(theta^2) bias).
  //  kLinearModel -- displace the normalized point by the linear flow of theta=log(dRCam),
  //                  so Stage 1 inverts it to theta exactly. Isolates convention/sign/timing
  //                  correctness.
  //  kRowTimedLinear -- RS-validator: each vector is exposed at its own row time, so the flow
  //                  is M*(omega0 + alpha*tau_i)*dt_i (an intra-frame angular-rate gradient
  //                  alpha). omega0 = log(dRCam)/dtCenter. The RS-aware fit recovers omega0
  //                  exactly; frame-center is biased by alpha. Requires setRowTiming().
  enum class Mode {
    kProjection,
    kLinearModel,
    kRowTimedLinear
  };

  struct PairSpec {
    double frameStartTimeA = 0.0;
    double frameStartTimeB = 0.0;
    Eigen::Quaterniond dRCam = Eigen::Quaterniond::Identity(); // camera-frame relative rotation
    Eigen::Vector3d alphaCam = Eigen::Vector3d::Zero(); // camera-frame angular accel (rad/s^2), row-timed mode
  };

  SyntheticFlowSource(const Intrinsics& intrinsics, int width, int height, int stride,
    Mode mode = Mode::kProjection) :
    m_intrinsics(intrinsics),
    m_width(width),
    m_height(height),
    m_stride(stride),
    m_mode(mode) {}

  void setPairs(std::vector<PairSpec> pairs) { m_pairs = std::move(pairs); }
  void setRowTiming(const RollingShutterTiming& timing) { m_timing = timing; }

  size_t pairCount() const override { return m_pairs.size(); }
  bool pair(size_t pairIndex, FramePairFlow& outPair) override;

private:
  const Intrinsics& m_intrinsics;
  int m_width, m_height, m_stride;
  Mode m_mode;
  RollingShutterTiming m_timing; // used by kRowTimedLinear
  std::vector<PairSpec> m_pairs;
};

} // namespace CameraImuCalib
