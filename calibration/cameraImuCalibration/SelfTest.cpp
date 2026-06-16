// Synthetic round-trip self-test for the camera-IMU autosync calibrator.
//
// Picks ground-truth R_imu_to_cam, delta, bias and a smooth band-limited body angular
// velocity exciting all three axes; synthesizes a gyro CSV (t_imu = t_cam + delta) and dense
// flow per frame pair; runs the full estimator; asserts recovery of R (< 0.05 deg), delta
// (< 0.1 ms), bias (< 1e-3 deg/s). Repeats with flipReadout = true and asserts the v1 result
// is unchanged (frame-center is flip-invariant). This is the primary guard against sign and
// axis-convention errors.

#include "Estimator.h"
#include "ImuTrace.h"
#include "Intrinsics.h"
#include "RollingShutterTiming.h"
#include "SyntheticFlowSource.h"
#include <Eigen/Geometry>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace CameraImuCalib;

namespace {

// Ground-truth body angular velocity, deg/s. Band-limited multi-tone exciting all three
// axes, with several-Hz content so that angular acceleration is non-trivial -- which is what
// makes the time offset delta observable. Held near zero for the first 2 s to provide a
// static segment for bias init.
Eigen::Vector3d omegaBodyDeg(double t) {
  if (t < 2.0)
    return Eigen::Vector3d::Zero();
  const double s = t - 2.0;
  const double twoPi = 2.0 * M_PI;
  const double wx = 22.0 * std::sin(twoPi * 1.3 * s) + 14.0 * std::sin(twoPi * 3.7 * s + 0.7) +
    9.0 * std::sin(twoPi * 6.1 * s + 1.9);
  const double wy = 19.0 * std::sin(twoPi * 1.7 * s + 0.5) + 12.0 * std::sin(twoPi * 4.3 * s + 1.3) +
    8.0 * std::sin(twoPi * 7.1 * s);
  const double wz = 17.0 * std::sin(twoPi * 1.1 * s + 2.0) + 11.0 * std::sin(twoPi * 3.1 * s + 0.4) +
    7.0 * std::sin(twoPi * 5.9 * s + 1.1);
  return Eigen::Vector3d(wx, wy, wz);
}

// Integrate the body-frame rotation over [tStart, tEnd] on the camera clock by composing
// fine exponential-map increments (ground-truth reference, independent of ImuTrace).
Eigen::Quaterniond integrateOmegaBody(double tStart, double tEnd, int substeps) {
  Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
  const double dt = (tEnd - tStart) / substeps;
  for (int k = 0; k < substeps; ++k) {
    const double tMid = tStart + (k + 0.5) * dt;
    const Eigen::Vector3d omegaRad = degToRad(omegaBodyDeg(tMid));
    q = q * expSO3<double>(Eigen::Vector3d(omegaRad * dt));
    q.normalize();
  }
  return q;
}

// Ground-truth body-frame angular acceleration (rad/s^2), central finite difference.
Eigen::Vector3d omegaBodyAlphaRad(double t) {
  const double h = 1.0e-3;
  return (degToRad(omegaBodyDeg(t + h)) - degToRad(omegaBodyDeg(t - h))) / (2.0 * h);
}

double quaternionAngleDeg(const Eigen::Quaterniond& a, const Eigen::Quaterniond& b) {
  double dot = std::abs(a.normalized().dot(b.normalized()));
  dot = std::min(1.0, std::max(-1.0, dot));
  return 2.0 * std::acos(dot) * (180.0 / M_PI);
}

} // namespace

int main(int /*argc*/, char** /*argv*/) {
  // ---- ground truth ----
  const Eigen::Quaterniond rGt(
    Eigen::AngleAxisd(degToRad(80.0), Eigen::Vector3d(0.2, 0.9, -0.3).normalized()));
  const double deltaGt = 0.003; // +3 ms
  const Eigen::Vector3d biasGt(0.6, -0.4, 0.9); // deg/s

  // ---- intrinsics + timing ----
  Intrinsics intrinsics;
  intrinsics.set(900.0, 900.0, 960.0, 540.0,
    {-0.006, -0.007, -0.0003, -0.001, 1.0e-4}, cv::Size(1920, 1080));
  RollingShutterTiming timing; // lineDelay = 1/112500, visibleHeight = 1080, flip = false

  const double fps = 90.0;
  const double duration = 10.0;
  const int frameCount = static_cast<int>(fps * duration);
  const double imuRateHz = 3600.0;

  // Camera and IMU share one physical clock starting at 0 (mirroring CalibrationWriter's
  // common base offset). Frames start at tFirstFrame so there is IMU history on both sides
  // of every integration window. gyro(s) = omega_body(s - deltaGt) + bias, so the only
  // camera<->IMU offset is deltaGt: the estimator should recover delta = deltaGt.
  const double tFirstFrame = 0.050;

  // ---- synthesize the gyro CSV ----
  const char* imuPath = "/tmp/cameraImuSelfTest_imu.csv";
  {
    FILE* f = fopen(imuPath, "w");
    if (!f) {
      fprintf(stderr, "self-test: cannot write %s\n", imuPath);
      return 2;
    }
    fprintf(f, "timestamp,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z\n");
    const double imuEnd = tFirstFrame + duration + 0.060; // cover the last window + margin
    const size_t nSamples = static_cast<size_t>(imuEnd * imuRateHz);
    for (size_t k = 0; k < nSamples; ++k) {
      const double s = static_cast<double>(k) / imuRateHz;
      const Eigen::Vector3d g = omegaBodyDeg(s - deltaGt) + biasGt;
      const long long ns = static_cast<long long>(std::llround(s * 1e9));
      fprintf(f, "%016lld,%f,%f,%f,%f,%f,%f\n", ns, g.x(), g.y(), g.z(), 0.0, 0.0, 1.0);
    }
    fclose(f);
  }

  ImuTrace imu;
  if (!imu.load(imuPath)) {
    fprintf(stderr, "self-test: ImuTrace load failed\n");
    return 2;
  }

  // ---- synthesize flow pairs from the ground-truth camera rotation per pair ----
  std::vector<SyntheticFlowSource::PairSpec> pairs;
  pairs.reserve(frameCount - 1);
  for (int i = 0; i + 1 < frameCount; ++i) {
    const double tA = tFirstFrame + static_cast<double>(i) / fps;
    const double tB = tFirstFrame + static_cast<double>(i + 1) / fps;
    const double tcA = timing.frameCenterTime(tA);
    const double tcB = timing.frameCenterTime(tB);
    const Eigen::Quaterniond dRBody = integrateOmegaBody(tcA, tcB, 32);
    SyntheticFlowSource::PairSpec spec;
    spec.frameStartTimeA = tA;
    spec.frameStartTimeB = tB;
    spec.dRCam = rGt * dRBody * rGt.conjugate(); // dR_cam = R * dR_imu * R^-1
    spec.dRCam.normalize();
    // Camera-frame angular acceleration at the pair center (for the row-timed validator).
    spec.alphaCam = rGt * omegaBodyAlphaRad(0.5 * (tcA + tcB));
    pairs.push_back(spec);
  }

  EstimatorConfig config; // defaults
  config.refineBias = true; // exercise the Ceres bias-refinement path (the stringent case);
    // production defaults to the fixed static-segment bias.

  auto run = [&](SyntheticFlowSource::Mode mode, bool flip, const EstimatorConfig& cfg,
               EstimatorResult& result) -> bool {
    RollingShutterTiming t = timing;
    t.flipReadout = flip;
    SyntheticFlowSource flow(intrinsics, 1920, 1080, /*stride=*/ 16, mode);
    flow.setPairs(pairs);
    flow.setRowTiming(t); // generation must use the same readout direction as estimation
    Estimator estimator(intrinsics, t, imu, cfg);
    return estimator.run(flow, result);
  };
  auto report = [&](const char* tag, const EstimatorResult& r) {
    printf("\n--- %s: recovery vs ground truth ---\n", tag);
    printf("R error            : %.5f deg   [est vs gt]\n", quaternionAngleDeg(r.rImuToCam, rGt));
    printf("delta error        : %.5f ms    [est %.4f ms, gt %.4f ms]\n",
      std::abs(r.deltaSeconds - deltaGt) * 1e3, r.deltaSeconds * 1e3, deltaGt * 1e3);
    printf("bias error (maxabs): %.6f deg/s [est (%.4f,%.4f,%.4f)]\n",
      (r.biasDeg - biasGt).cwiseAbs().maxCoeff(), r.biasDeg.x(), r.biasDeg.y(), r.biasDeg.z());
    printf("residual RMS       : %.5f deg   delta_coarse %.4f ms (peak %.3f)\n",
      r.residualRmsDeg, r.deltaCoarseSeconds * 1e3, r.stage2CorrelationPeak);
  };

  bool pass = true;
  auto check = [&](bool ok, const char* what) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    pass = pass && ok;
  };

  using Mode = SyntheticFlowSource::Mode;

  // ---- Pass 1: linear-model flow -- the strict convention/sign/timing guard ----
  printf("\n===== self-test: linear-model flow (convention guard) =====\n");
  EstimatorResult lin;
  if (!run(Mode::kLinearModel, false, config, lin)) {
    fprintf(stderr, "self-test: estimator failed\n");
    return 2;
  }
  report("linear-model", lin);

  // ---- Pass 2: flip-invariance (v1 is flip-invariant) ----
  printf("\n===== self-test: linear-model flow, flipReadout = true (invariance) =====\n");
  EstimatorResult linFlip;
  if (!run(Mode::kLinearModel, true, config, linFlip)) {
    fprintf(stderr, "self-test: estimator (flip) failed\n");
    return 2;
  }
  const double flipR = quaternionAngleDeg(lin.rImuToCam, linFlip.rImuToCam);
  const double flipDeltaMs = std::abs(lin.deltaSeconds - linFlip.deltaSeconds) * 1e3;
  const double flipBias = (lin.biasDeg - linFlip.biasDeg).cwiseAbs().maxCoeff();
  printf("flip vs no-flip: dR = %.6f deg, ddelta = %.6f ms, dbias = %.6f deg/s\n",
    flipR, flipDeltaMs, flipBias);

  // ---- Pass 3: projection flow -- exercises the full undistortion path (realistic tols) ----
  printf("\n===== self-test: projection flow (undistortion path) =====\n");
  EstimatorResult proj;
  if (!run(Mode::kProjection, false, config, proj)) {
    fprintf(stderr, "self-test: estimator (proj) failed\n");
    return 2;
  }
  report("projection", proj);

  // ---- Pass 4: row-timed flow -- RS-aware fit vs frame-center on the SAME data ----
  // Row-timed flow carries an intra-frame angular-rate gradient (alpha) that frame-center
  // cannot fit. The RS-aware (omega0 + alpha) fit should recover the ground truth exactly,
  // while frame-center on the same data is biased.
  printf("\n===== self-test: row-timed flow (rolling-shutter-aware) =====\n");
  EstimatorConfig configRS = config;
  configRS.rollingShutterAware = true;
  configRS.rsAlphaRegularization = 0.0; // validate the RS math without the prior
  EstimatorConfig configRSbig = config; // huge prior -> alpha suppressed -> matches frame-center
  configRSbig.rollingShutterAware = true;
  configRSbig.rsAlphaRegularization = 1.0e6;
  EstimatorResult rowRS, rowFC, rowRSbig;
  if (!run(Mode::kRowTimedLinear, false, configRS, rowRS) ||
    !run(Mode::kRowTimedLinear, false, config, rowFC) ||
    !run(Mode::kRowTimedLinear, false, configRSbig, rowRSbig)) {
    fprintf(stderr, "self-test: estimator (row-timed) failed\n");
    return 2;
  }
  report("row-timed RS-aware", rowRS);
  report("row-timed frame-center", rowFC);
  const double rowRSerr = quaternionAngleDeg(rowRS.rImuToCam, rGt);
  const double rowFCerr = quaternionAngleDeg(rowFC.rImuToCam, rGt);
  // Graceful degradation: a large alpha prior collapses RS-aware onto the frame-center result
  // (alpha forced to ~0), so enabling --rolling-shutter can never wildly overfit.
  const double regVsFc = quaternionAngleDeg(rowRSbig.rImuToCam, rowFC.rImuToCam);
  printf("row-timed: RS-aware R err %.5f deg vs frame-center R err %.5f deg "
         "(inlier %.3f vs %.3f); large-prior RS vs frame-center dR = %.5f deg\n",
    rowRSerr, rowFCerr, rowRS.medianInlierRatio, rowFC.medianInlierRatio, regVsFc);

  // ---- verdict ----
  printf("\n--- verdict ---\n");
  // Strict tolerances apply to the linear-model pass (no linearization bias).
  check(quaternionAngleDeg(lin.rImuToCam, rGt) < 0.05, "[linear] R recovered within 0.05 deg");
  check(std::abs(lin.deltaSeconds - deltaGt) * 1e3 < 0.1, "[linear] delta recovered within 0.1 ms");
  check((lin.biasDeg - biasGt).cwiseAbs().maxCoeff() < 1e-3, "[linear] bias recovered within 1e-3 deg/s");
  check(flipR < 1e-6 && flipDeltaMs < 1e-6 && flipBias < 1e-6, "[linear] flip-invariance (v1 unchanged)");
  // Projection pass carries the expected O(theta^2) linear-flow bias -> looser, realistic tols.
  check(quaternionAngleDeg(proj.rImuToCam, rGt) < 0.25, "[projection] R within 0.25 deg (undistortion path)");
  check(std::abs(proj.deltaSeconds - deltaGt) * 1e3 < 0.2, "[projection] delta within 0.2 ms");
  check((proj.biasDeg - biasGt).cwiseAbs().maxCoeff() < 0.1, "[projection] bias within 0.1 deg/s");
  // RS-aware recovers row-timed flow exactly, and beats frame-center on the same data.
  check(rowRSerr < 0.05, "[row-timed] RS-aware R within 0.05 deg");
  check(std::abs(rowRS.deltaSeconds - deltaGt) * 1e3 < 0.1, "[row-timed] RS-aware delta within 0.1 ms");
  check((rowRS.biasDeg - biasGt).cwiseAbs().maxCoeff() < 1e-3, "[row-timed] RS-aware bias within 1e-3 deg/s");
  check(rowRSerr < rowFCerr, "[row-timed] RS-aware beats frame-center on row-timed flow");
  // Large prior should pull RS-aware most of the way back to frame-center: far closer to it
  // than the un-regularized RS result is (which differs from frame-center by rowFCerr).
  check(regVsFc < 0.25 * rowFCerr, "[row-timed] large alpha prior degrades RS-aware to frame-center");

  printf("\n%s\n", pass ? "SELF-TEST PASSED" : "SELF-TEST FAILED");
  return pass ? 0 : 1;
}
