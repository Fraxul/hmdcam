#pragma once
#include "Conventions.h"
#include "FlowSource.h"
#include "ImuTrace.h"
#include "Intrinsics.h"
#include "RollingShutterTiming.h"
#include <Eigen/Geometry>
#include <string>
#include <vector>

namespace CameraImuCalib {

// Tunables for the six-stage pipeline.
struct EstimatorConfig {
  // Stage 1 robust flow fit.
  double stage1CauchyPixels = 2.0; // Cauchy scale, pixels (converted to normalized via fx)
  int stage1IrlsIterations = 10;
  double maxPairAngleDeg = 12.0; // drop pairs above this (linear-flow model invalid)
  double thetaMinDeg = 0.5; // min per-pair rotation to carry directional info
  double inlierRatioThreshold = 0.6; // drop low-inlier pairs
  // Row-timed rolling-shutter-aware flow fit: time each flow vector by its row's exposure
  // (t_row) instead of the frame center. Removes the intra-frame timing bias that caps the
  // inlier ratio at high angular rates. Uses the timing model's flip flag.
  bool rollingShutterAware = false;
  // Tikhonov prior on the RS-aware angular-acceleration term alpha, as a multiple of the
  // alpha-block's own scale (scale-invariant). alpha is pulled toward 0 unless the data
  // supports it, so enabling --rolling-shutter degrades gracefully to frame-center when the
  // motion has no real intra-frame gradient (rather than overfitting flow noise / row-correlated
  // parallax into alpha). 0 = pure RS-aware; larger = closer to frame-center.
  double rsAlphaRegularization = 1.0;

  // Stage 2 coarse time offset.
  double timeSearchHalfWindowSec = 0.050; // ±50 ms
  double resampleRateHz = 1000.0;

  // Stage 3 bias init.
  // By default the gyro bias is taken from the static segment (gyro-at-rest) and held fixed --
  // the grounded, reliable choice. Setting refineBias lets Ceres adjust bias during motion,
  // which can be more accurate on a clean capture but tends to absorb parallax/RS model error
  // into bias on contaminated ones (so it is opt-in, gated behind --refine-bias).
  bool refineBias = false;
  // Threshold for "not rotating" when seeking a static segment for bias init. Must sit ABOVE
  // the gyro bias magnitude (observed ~1.8 deg/s on this rig) yet well below real motion
  // (tens of deg/s), so the at-rest segment registers as static. Bias is refined by Ceres
  // regardless, so this only affects the initializer.
  double staticBiasMaxRateDegPerSec = 5.0;
  double staticBiasMinDurationSec = 0.5;

  // Stage 5 refinement.
  double stage5CauchyRadians = 0.005; // ~0.3 deg
  int refineMaxIterations = 100;
  bool weightByInlierRatio = true;

  // Diagnostics / limits.
  size_t maxPairs = 0; // 0 = all pairs; otherwise cap Stage-1 pairs (for quick tests)
  size_t progressInterval = 250; // log Stage-1 progress every N pairs (0 = silent)
};

// Per-pair plotting record: camera-frame angular velocity from flow vs the
// IMU prediction R*(omega_imu - b), both rad/s, at the pair midpoint camera time.
struct PairPlotRecord {
  double tMid = 0.0;
  Eigen::Vector3d omegaCam = Eigen::Vector3d::Zero();
  Eigen::Vector3d omegaImuPredicted = Eigen::Vector3d::Zero();
  double omegaCamMag = 0.0;
  double omegaImuMag = 0.0;
};

struct EstimatorResult {
  Eigen::Quaterniond rImuToCam = Eigen::Quaterniond::Identity(); // v_cam = R * v_imu
  double deltaSeconds = 0.0; // refined; t_imu = t_cam + delta
  double deltaCoarseSeconds = 0.0; // Stage 2 cross-correlation result
  Eigen::Vector3d biasDeg = Eigen::Vector3d::Zero();
  bool biasRefined = false;

  double residualRmsDeg = 0.0;
  Eigen::Vector3d perAxisResidualRmsDeg = Eigen::Vector3d::Zero();
  size_t pairsUsed = 0;
  size_t pairsRejected = 0;
  double medianInlierRatio = 0.0;
  double stage2CorrelationPeak = 0.0;

  std::vector<PairPlotRecord> plot; // ordered by tMid
  std::vector<std::string> warnings;
  bool converged = false;
};

class Estimator {
public:
  Estimator(const Intrinsics& intrinsics, const RollingShutterTiming& timing,
    const ImuTrace& imuTrace, const EstimatorConfig& config) :
    m_intrinsics(intrinsics),
    m_timing(timing),
    m_imu(imuTrace),
    m_config(config) {}

  // Run stages 1-6 over the supplied flow source. Returns false only on fatal failure
  // (no usable pairs); soft problems are surfaced as result.warnings.
  bool run(FlowSource& flowSource, EstimatorResult& outResult);

private:
  // One frame pair after Stage 1: the flow-derived relative rotation and its health.
  struct PairMeasurement {
    double tCenterA = 0.0;
    double tCenterB = 0.0;
    double tMid = 0.0;
    double dt = 0.0;
    Eigen::Vector3d thetaCam = Eigen::Vector3d::Zero(); // relative-rotation vector (rad)
    double angleDeg = 0.0;
    double inlierRatio = 0.0;
    bool usableForRotation = false; // passes angle gate, thetaMin, inlier threshold
  };

  bool stage1BuildMeasurements(FlowSource& flowSource);
  bool fitPairRotation(const FramePairFlow& pair, Eigen::Vector3d& outTheta,
    double& outInlierRatio) const;
  double stage2CoarseTimeOffset(double& outPeak) const;
  void stage3InitRotationAndBias(double deltaCoarse, Eigen::Quaterniond& outR,
    Eigen::Vector3d& outBiasDeg, bool& outStaticBiasFound) const;

  const Intrinsics& m_intrinsics;
  const RollingShutterTiming& m_timing;
  const ImuTrace& m_imu;
  EstimatorConfig m_config;

  std::vector<PairMeasurement> m_pairs; // all pairs (for magnitude series + rotation)
};

} // namespace CameraImuCalib
