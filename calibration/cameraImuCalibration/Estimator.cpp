#include "Estimator.h"
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow" // Ignore warning in ceres/dynamic_autodiff_cost_function.h
#include <ceres/ceres.h>
#pragma clang diagnostic pop
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace CameraImuCalib {

namespace {

  // Linear-interpolating lookup into a sorted (time, value) series. Clamps at the ends.
  double interpSeries(const std::vector<double>& times, const std::vector<double>& values,
    double t) {
    if (times.empty())
      return 0.0;
    if (t <= times.front())
      return values.front();
    if (t >= times.back())
      return values.back();
    const size_t hi = static_cast<size_t>(
      std::upper_bound(times.begin(), times.end(), t) - times.begin());
    const size_t lo = hi - 1;
    const double frac = (t - times[lo]) / (times[hi] - times[lo]);
    return values[lo] + frac * (values[hi] - values[lo]);
  }

  // Ceres autodiff residual for one frame pair:
  //   dR_imu      = integrate gyro over [t_center_i + delta, t_center_{i+1} + delta], bias b
  //   predicted   = R_imu_to_cam * dR_imu * R_imu_to_cam^-1
  //   r           = Log( R_cam_measured^-1 * predicted )   (3-vector, scaled by sqrt weight)
  //
  // Sign note: we adopt t_imu = t_cam + delta (the documented output relation), so the
  // IMU-clock window is the camera-clock interval shifted by +delta. The synthetic self-test
  // generates data under this same physical relation and confirms the sign and magnitude.
  struct PairResidual {
    PairResidual(const ImuTrace* imu, double tCenterA, double tCenterB,
      const Eigen::Vector3d& thetaCam, double weight) :
      m_imu(imu),
      m_tCenterA(tCenterA),
      m_tCenterB(tCenterB),
      m_thetaCam(thetaCam),
      m_weight(weight) {}

    template <typename T>
    bool operator()(const T* const qCoeffs, const T* const delta, const T* const biasDeg,
      T* residual) const {
      // qCoeffs is Eigen quaternion storage order [x, y, z, w] (EigenQuaternionManifold).
      const Eigen::Quaternion<T> rImuToCam(qCoeffs[3], qCoeffs[0], qCoeffs[1], qCoeffs[2]);
      const T t0 = T(m_tCenterA) + delta[0];
      const T t1 = T(m_tCenterB) + delta[0];
      const Eigen::Matrix<T, 3, 1> bias(biasDeg[0], biasDeg[1], biasDeg[2]);

      const Eigen::Quaternion<T> dRImu = m_imu->integrateBodyRotation(t0, t1, bias);
      const Eigen::Quaternion<T> predicted = transformRelativeRotation(rImuToCam, dRImu);

      const Eigen::Matrix<T, 3, 1> thetaCamT = m_thetaCam.cast<T>();
      const Eigen::Quaternion<T> measured = expSO3(thetaCamT);
      const Eigen::Matrix<T, 3, 1> r = logSO3(Eigen::Quaternion<T>(measured.conjugate() * predicted));

      residual[0] = T(m_weight) * r[0];
      residual[1] = T(m_weight) * r[1];
      residual[2] = T(m_weight) * r[2];
      return true;
    }

    const ImuTrace* m_imu;
    double m_tCenterA, m_tCenterB;
    Eigen::Vector3d m_thetaCam;
    double m_weight;
  };

} // namespace

// ---- Stage 1: dense flow -> per-pair camera relative rotation -----------------------

bool Estimator::fitPairRotation(const FramePairFlow& pair, Eigen::Vector3d& outTheta,
  double& outInlierRatio) const {
  const size_t n = pair.samples.size();
  if (n < 8) // need a handful of points to constrain 3 DoF
    return false;

  // Undistort both endpoints to normalized camera coordinates in one batch each.
  std::vector<cv::Point2d> px0(n), px1(n), n0, n1;
  for (size_t i = 0; i < n; ++i) {
    px0[i] = cv::Point2d(pair.samples[i].pixel0.x(), pair.samples[i].pixel0.y());
    px1[i] = cv::Point2d(pair.samples[i].pixel1.x(), pair.samples[i].pixel1.y());
  }
  m_intrinsics.undistortToNormalized(px0, n0);
  m_intrinsics.undistortToNormalized(px1, n1);

  // Cauchy scale in normalized units: pixels / focal length.
  const double cauchy = m_config.stage1CauchyPixels / m_intrinsics.fx();
  const double cauchySq = cauchy * cauchy;
  const double dtCenter = pair.frameStartTimeB - pair.frameStartTimeA;

  if (!m_config.rollingShutterAware) {
    // ---- v1 frame-center: 3-DoF fit for the rotation vector theta = omega * dtCenter ----
    // Rows per sample:  rowx . theta = xdot ;  rowy . theta = ydot
    Eigen::Vector3d theta = Eigen::Vector3d::Zero();
    std::vector<double> weight(n, 1.0);
    for (int iter = 0; iter < m_config.stage1IrlsIterations; ++iter) {
      Eigen::Matrix3d ata = Eigen::Matrix3d::Zero();
      Eigen::Vector3d atb = Eigen::Vector3d::Zero();
      for (size_t i = 0; i < n; ++i) {
        const double x = n0[i].x, y = n0[i].y;
        const double xdot = n1[i].x - n0[i].x, ydot = n1[i].y - n0[i].y;
        const Eigen::RowVector3d rowx(x * y, -(1.0 + x * x), y);
        const Eigen::RowVector3d rowy(1.0 + y * y, -x * y, -x);
        const double w = weight[i];
        ata += w * (rowx.transpose() * rowx + rowy.transpose() * rowy);
        atb += w * (rowx.transpose() * xdot + rowy.transpose() * ydot);
      }
      const Eigen::Vector3d next = ata.ldlt().solve(atb);
      if (!next.allFinite())
        return false;
      theta = next;
      for (size_t i = 0; i < n; ++i) {
        const double x = n0[i].x, y = n0[i].y;
        const double xdot = n1[i].x - n0[i].x, ydot = n1[i].y - n0[i].y;
        const double rx = (x * y) * theta[0] - (1.0 + x * x) * theta[1] + y * theta[2] - xdot;
        const double ry = (1.0 + y * y) * theta[0] - (x * y) * theta[1] - x * theta[2] - ydot;
        weight[i] = pair.samples[i].confidence / (1.0 + (rx * rx + ry * ry) / cauchySq);
      }
    }
    size_t inliers = 0;
    for (size_t i = 0; i < n; ++i) {
      const double x = n0[i].x, y = n0[i].y;
      const double xdot = n1[i].x - n0[i].x, ydot = n1[i].y - n0[i].y;
      const double rx = (x * y) * theta[0] - (1.0 + x * x) * theta[1] + y * theta[2] - xdot;
      const double ry = (1.0 + y * y) * theta[0] - (x * y) * theta[1] - x * theta[2] - ydot;
      if ((rx * rx + ry * ry) <= cauchySq)
        ++inliers;
    }
    outInlierRatio = static_cast<double>(inliers) / static_cast<double>(n);
    outTheta = theta;
    return true;
  }

  // ---- Rolling-shutter-aware: 6-DoF fit for [omega0, alpha] -----------------------------
  // Each vector is exposed near its own time tau_i (row offset from the pair center, within
  // +/- the readout half-time). Model the camera angular velocity as omega(t) = omega0 +
  // alpha*(t - tMidPair); a vector's flow is then M(x,y) * (omega0 + alpha*tau_i) * dt_i.
  // Stacking start/end rows gives a 6-DoF linear system; alpha absorbs the intra-frame
  // angular-rate gradient that frame-center cannot fit. theta_cam = omega0 * dtCenter (the
  // time-varying part integrates to zero over the symmetric frame-center interval).
  const double tMidPair = 0.5 * (m_timing.frameCenterTime(pair.frameStartTimeA) + m_timing.frameCenterTime(pair.frameStartTimeB));
  const int height = static_cast<int>(m_timing.visibleHeight);
  std::vector<double> dt(n), tau(n);
  for (size_t i = 0; i < n; ++i) {
    const int y1 = std::min(std::max(0, static_cast<int>(std::lround(pair.samples[i].pixel0.y()))), height - 1);
    const int y2 = std::min(std::max(0, static_cast<int>(std::lround(pair.samples[i].pixel1.y()))), height - 1);
    const double ta = m_timing.rowExposureTime(pair.frameStartTimeA, y1);
    const double tb = m_timing.rowExposureTime(pair.frameStartTimeB, y2);
    dt[i] = (tb - ta > 1e-6) ? (tb - ta) : dtCenter;
    tau[i] = 0.5 * (ta + tb) - tMidPair;
  }

  using Vector6 = Eigen::Matrix<double, 6, 1>;
  using Matrix6 = Eigen::Matrix<double, 6, 6>;
  Vector6 u = Vector6::Zero(); // [omega0 (0:3), alpha (3:6)]
  std::vector<double> weight(n, 1.0);
  auto designRows = [&](size_t i, Vector6& ax, Vector6& ay) {
    const double x = n0[i].x, y = n0[i].y;
    const Eigen::Vector3d mx(x * y, -(1.0 + x * x), y);
    const Eigen::Vector3d my(1.0 + y * y, -x * y, -x);
    ax.head<3>() = dt[i] * mx;
    ax.tail<3>() = dt[i] * tau[i] * mx;
    ay.head<3>() = dt[i] * my;
    ay.tail<3>() = dt[i] * tau[i] * my;
  };
  for (int iter = 0; iter < m_config.stage1IrlsIterations; ++iter) {
    Matrix6 ata = Matrix6::Zero();
    Vector6 atb = Vector6::Zero();
    for (size_t i = 0; i < n; ++i) {
      Vector6 ax, ay;
      designRows(i, ax, ay);
      const double xdot = n1[i].x - n0[i].x, ydot = n1[i].y - n0[i].y;
      const double w = weight[i];
      ata += w * (ax * ax.transpose() + ay * ay.transpose());
      atb += w * (ax * xdot + ay * ydot);
    }
    // Tikhonov prior on the alpha block (last 3 params): add a ridge scaled to the block's own
    // diagonal, so alpha is shrunk toward 0 unless the data strongly supports it. This makes
    // RS-aware degrade gracefully to frame-center when there is no real intra-frame gradient.
    if (m_config.rsAlphaRegularization > 0.0) {
      const double alphaDiag = (ata(3, 3) + ata(4, 4) + ata(5, 5)) / 3.0;
      const double ridge = m_config.rsAlphaRegularization * alphaDiag;
      ata(3, 3) += ridge;
      ata(4, 4) += ridge;
      ata(5, 5) += ridge;
    }
    const Vector6 next = ata.ldlt().solve(atb);
    if (!next.allFinite())
      return false;
    u = next;
    for (size_t i = 0; i < n; ++i) {
      Vector6 ax, ay;
      designRows(i, ax, ay);
      const double xdot = n1[i].x - n0[i].x, ydot = n1[i].y - n0[i].y;
      const double rx = ax.dot(u) - xdot, ry = ay.dot(u) - ydot;
      weight[i] = pair.samples[i].confidence / (1.0 + (rx * rx + ry * ry) / cauchySq);
    }
  }
  size_t inliers = 0;
  for (size_t i = 0; i < n; ++i) {
    Vector6 ax, ay;
    designRows(i, ax, ay);
    const double xdot = n1[i].x - n0[i].x, ydot = n1[i].y - n0[i].y;
    const double rx = ax.dot(u) - xdot, ry = ay.dot(u) - ydot;
    if ((rx * rx + ry * ry) <= cauchySq)
      ++inliers;
  }
  outInlierRatio = static_cast<double>(inliers) / static_cast<double>(n);
  outTheta = u.head<3>() * dtCenter; // omega0 * dtCenter
  return true;
}

bool Estimator::stage1BuildMeasurements(FlowSource& flowSource) {
  m_pairs.clear();
  const size_t pairCount = flowSource.pairCount();
  size_t droppedAngle = 0;

  const size_t limit = m_config.maxPairs ? std::min(m_config.maxPairs, pairCount) : pairCount;
  FramePairFlow pair;
  for (size_t p = 0; p < limit; ++p) {
    if (m_config.progressInterval && (p % m_config.progressInterval == 0))
      printf("Stage 1: processing pair %zu/%zu...\n", p, limit);
    if (!flowSource.pair(p, pair))
      continue;

    PairMeasurement m;
    m.tCenterA = m_timing.frameCenterTime(pair.frameStartTimeA);
    m.tCenterB = m_timing.frameCenterTime(pair.frameStartTimeB);
    m.dt = m.tCenterB - m.tCenterA;
    m.tMid = 0.5 * (m.tCenterA + m.tCenterB);
    if (m.dt <= 0.0)
      continue;

    if (!fitPairRotation(pair, m.thetaCam, m.inlierRatio))
      continue;

    m.angleDeg = radToDeg(m.thetaCam.norm());

    // Linear-flow model validity gate: drop pairs whose inter-frame rotation is too
    // large to linearize (e.g. long gaps from writeInterval/drops). Logged, not silent.
    if (m.angleDeg > m_config.maxPairAngleDeg) {
      ++droppedAngle;
      m.usableForRotation = false;
    } else {
      m.usableForRotation = (m.angleDeg >= m_config.thetaMinDeg) &&
        (m.inlierRatio >= m_config.inlierRatioThreshold);
    }
    m_pairs.push_back(m);
  }

  if (droppedAngle)
    printf("Stage 1: dropped %zu pairs exceeding the %.1f deg linear-model gate "
           "(large inter-frame rotation)\n",
      droppedAngle, m_config.maxPairAngleDeg);
  printf("Stage 1: %zu/%zu frame pairs produced a rotation estimate\n", m_pairs.size(), pairCount);
  return !m_pairs.empty();
}

// ---- Stage 2: coarse time offset via magnitude cross-correlation --------------------

double Estimator::stage2CoarseTimeOffset(double& outPeak) const {
  // Camera magnitude series at pair midpoints; IMU magnitude series at sample times. Both
  // rad/s. Magnitude is invariant to R_imu_to_cam and (at motion rates) to bias.
  std::vector<double> camT, camMag;
  camT.reserve(m_pairs.size());
  camMag.reserve(m_pairs.size());
  for (const PairMeasurement& m : m_pairs) {
    camT.push_back(m.tMid);
    camMag.push_back(m.thetaCam.norm() / m.dt);
  }

  std::vector<double> imuT, imuMag;
  imuT.reserve(m_imu.sampleCount());
  imuMag.reserve(m_imu.sampleCount());
  for (size_t i = 0; i < m_imu.sampleCount(); ++i) {
    imuT.push_back(m_imu.sample(i).time);
    imuMag.push_back(degToRad(m_imu.sample(i).gyroDeg.norm()));
  }

  // Uniform grid over the camera-time span, shrunk so that t+delta stays inside the IMU
  // span for every tested delta.
  const double w = m_config.timeSearchHalfWindowSec;
  const double gridStart = std::max(camT.front(), m_imu.firstTime() - w) + w;
  const double gridEnd = std::min(camT.back(), m_imu.lastTime() + w) - w;
  if (gridEnd <= gridStart) {
    outPeak = 0.0;
    return 0.0;
  }
  const double dt = 1.0 / m_config.resampleRateHz;
  const size_t gridN = static_cast<size_t>((gridEnd - gridStart) / dt) + 1;

  std::vector<double> camGrid(gridN);
  for (size_t k = 0; k < gridN; ++k)
    camGrid[k] = interpSeries(camT, camMag, gridStart + k * dt);

  // Zero-mean normalized cross-correlation of camGrid(t) with imu(t + delta).
  auto correlationAt = [&](double delta) -> double {
    double sx = 0, sy = 0;
    for (size_t k = 0; k < gridN; ++k) {
      sx += camGrid[k];
      sy += interpSeries(imuT, imuMag, gridStart + k * dt + delta);
    }
    const double mx = sx / gridN, my = sy / gridN;
    double num = 0, dx = 0, dy = 0;
    for (size_t k = 0; k < gridN; ++k) {
      const double a = camGrid[k] - mx;
      const double b = interpSeries(imuT, imuMag, gridStart + k * dt + delta) - my;
      num += a * b;
      dx += a * a;
      dy += b * b;
    }
    const double den = std::sqrt(dx * dy);
    return den > 1e-12 ? num / den : 0.0;
  };

  // Evaluate the correlation across the lag window, then take the discrete argmax.
  const size_t lagN = static_cast<size_t>((2.0 * w) / dt) + 1;
  std::vector<double> corr(lagN);
  size_t bestK = 0;
  for (size_t k = 0; k < lagN; ++k) {
    corr[k] = correlationAt(-w + k * dt);
    if (corr[k] > corr[bestK])
      bestK = k;
  }
  double bestDelta = -w + bestK * dt;
  const double bestCorr = corr[bestK];

  // Parabolic sub-grid refinement using the immediate neighbors of the discrete peak.
  double refined = bestDelta;
  if (bestK > 0 && bestK + 1 < lagN) {
    const double cm = corr[bestK - 1], c0 = corr[bestK], cp = corr[bestK + 1];
    const double denom = cm - 2.0 * c0 + cp;
    if (std::abs(denom) > 1e-12) {
      const double shift = 0.5 * (cm - cp) / denom;
      if (std::abs(shift) <= 1.0)
        refined = bestDelta + shift * dt;
    }
  }

  outPeak = bestCorr;
  return refined;
}

// ---- Stage 3: initialize R_imu_to_cam (Kabsch) and bias -----------------------------

void Estimator::stage3InitRotationAndBias(double deltaCoarse, Eigen::Quaterniond& outR,
  Eigen::Vector3d& outBiasDeg, bool& outStaticBiasFound) const {
  // Bias from the longest static segment; zero if none. This is the bias used directly
  // by default (held fixed), or the initializer when --refine-bias lets Ceres adjust it.
  outBiasDeg.setZero();
  outStaticBiasFound = m_imu.findStaticBias(m_config.staticBiasMaxRateDegPerSec,
    m_config.staticBiasMinDurationSec, outBiasDeg);
  if (!outStaticBiasFound)
    printf("Stage 3: no static segment found; bias %s = 0\n",
      m_config.refineBias ? "init" : "(held fixed)");

  // Wahba/Kabsch: R minimizing sum || theta_cam - R * theta_imu ||^2 over usable pairs.
  // theta_imu integrated over the (coarse-)shifted window with zero bias for init.
  Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();
  size_t used = 0;
  for (const PairMeasurement& m : m_pairs) {
    if (!m.usableForRotation)
      continue;
    const double t0 = m.tCenterA + deltaCoarse;
    const double t1 = m.tCenterB + deltaCoarse;
    const Eigen::Quaterniond dRImu =
      m_imu.integrateBodyRotation<double>(t0, t1, Eigen::Vector3d::Zero());
    const Eigen::Vector3d thetaImu = logSO3(dRImu);
    if (radToDeg(thetaImu.norm()) < m_config.thetaMinDeg)
      continue;
    cross += m.thetaCam * thetaImu.transpose(); // M = sum b a^T, b=theta_cam, a=theta_imu
    ++used;
  }

  if (used < 3) {
    printf("Stage 3: only %zu usable pairs for Kabsch; R init = identity\n", used);
    outR = Eigen::Quaterniond::Identity();
    return;
  }

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(cross, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d u = svd.matrixU();
  Eigen::Matrix3d v = svd.matrixV();
  Eigen::Matrix3d d = Eigen::Matrix3d::Identity();
  d(2, 2) = (u * v.transpose()).determinant() < 0.0 ? -1.0 : 1.0; // reflection fix
  const Eigen::Matrix3d rMat = u * d * v.transpose();
  outR = Eigen::Quaterniond(rMat);
  outR.normalize();
  printf("Stage 3: Kabsch over %zu pairs\n", used);
}

// ---- run: orchestrate stages 1-6 ----------------------------------------------------

bool Estimator::run(FlowSource& flowSource, EstimatorResult& out) {
  if (!stage1BuildMeasurements(flowSource)) {
    fprintf(stderr, "Estimator: Stage 1 produced no usable pairs\n");
    return false;
  }

  // Median inlier ratio diagnostic.
  {
    std::vector<double> ratios;
    ratios.reserve(m_pairs.size());
    for (const PairMeasurement& m : m_pairs)
      ratios.push_back(m.inlierRatio);
    std::sort(ratios.begin(), ratios.end());
    out.medianInlierRatio = ratios[ratios.size() / 2];
  }

  double peak = 0.0;
  out.deltaCoarseSeconds = stage2CoarseTimeOffset(peak);
  out.stage2CorrelationPeak = peak;
  printf("Stage 2: delta_coarse = %.3f ms (correlation peak %.3f)\n",
    out.deltaCoarseSeconds * 1e3, peak);

  Eigen::Quaterniond rInit;
  Eigen::Vector3d biasInit;
  bool staticBiasFound = false;
  stage3InitRotationAndBias(out.deltaCoarseSeconds, rInit, biasInit, staticBiasFound);

  // ---- Stage 4/5: assemble and solve the joint refinement -----------------------------
  double q[4] = {rInit.x(), rInit.y(), rInit.z(), rInit.w()}; // Eigen storage [x,y,z,w]
  double delta = out.deltaCoarseSeconds;
  double bias[3] = {biasInit.x(), biasInit.y(), biasInit.z()};

  ceres::Problem problem;
  size_t used = 0;
  for (const PairMeasurement& m : m_pairs) {
    if (!m.usableForRotation)
      continue;
    const double weight = m_config.weightByInlierRatio ? std::sqrt(m.inlierRatio) : 1.0;
    auto* cost = new ceres::AutoDiffCostFunction<PairResidual, 3, 4, 1, 3>(
      new PairResidual(&m_imu, m.tCenterA, m.tCenterB, m.thetaCam, weight));
    problem.AddResidualBlock(cost, new ceres::CauchyLoss(m_config.stage5CauchyRadians),
      q, &delta, bias);
    ++used;
  }
  out.pairsUsed = used;
  out.pairsRejected = m_pairs.size() - used;

  if (used < 3) {
    fprintf(stderr, "Estimator: only %zu usable pairs for refinement\n", used);
    return false;
  }

  problem.SetManifold(q, new ceres::EigenQuaternionManifold);
  if (!m_config.refineBias)
    problem.SetParameterBlockConstant(bias); // hold bias at the static-segment value (default)

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.max_num_iterations = m_config.refineMaxIterations;
  options.function_tolerance = 1e-10;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  printf("Stage 5: %s (%d iterations, final cost %.3e)\n",
    summary.IsSolutionUsable() ? "converged" : "FAILED",
    static_cast<int>(summary.iterations.size()), summary.final_cost);

  out.converged = summary.IsSolutionUsable();
  out.rImuToCam = Eigen::Quaterniond(q[3], q[0], q[1], q[2]); // (w,x,y,z) <- storage [x,y,z,w]
  out.rImuToCam.normalize();
  out.deltaSeconds = delta;
  out.biasDeg = Eigen::Vector3d(bias[0], bias[1], bias[2]);
  out.biasRefined = m_config.refineBias;

  // ---- Stage 6: residuals, plotting series, and diagnostics -------------
  Eigen::Vector3d sqAccum = Eigen::Vector3d::Zero();
  size_t residualCount = 0;
  const Eigen::Matrix3d rMat = out.rImuToCam.toRotationMatrix();
  out.plot.clear();
  for (const PairMeasurement& m : m_pairs) {
    PairPlotRecord rec;
    rec.tMid = m.tMid;
    rec.omegaCam = m.thetaCam / m.dt;
    const Eigen::Vector3d gyroMid =
      m_imu.gyroDegAt<double>(m.tMid + out.deltaSeconds);
    rec.omegaImuPredicted = rMat * degToRad(gyroMid - out.biasDeg);
    rec.omegaCamMag = rec.omegaCam.norm();
    rec.omegaImuMag = degToRad(gyroMid.norm());
    out.plot.push_back(rec);

    if (!m.usableForRotation)
      continue;
    const Eigen::Quaterniond dRImu = m_imu.integrateBodyRotation<double>(
      m.tCenterA + out.deltaSeconds, m.tCenterB + out.deltaSeconds, out.biasDeg);
    const Eigen::Quaterniond predicted(rMat * dRImu.toRotationMatrix() * rMat.transpose());
    const Eigen::Quaterniond measured = expSO3<double>(m.thetaCam);
    const Eigen::Vector3d r = logSO3(Eigen::Quaterniond(measured.conjugate() * predicted));
    sqAccum += r.cwiseProduct(r);
    ++residualCount;
  }
  if (residualCount) {
    out.perAxisResidualRmsDeg =
      (sqAccum / static_cast<double>(residualCount)).cwiseSqrt() * (180.0 / M_PI);
    out.residualRmsDeg =
      std::sqrt(sqAccum.sum() / static_cast<double>(residualCount)) * (180.0 / M_PI);
  }

  // acceptance checks -> loud warnings.
  const double kFramePeriodMs = 1000.0 / 90.0;
  if (std::abs(out.deltaSeconds) * 1e3 > kFramePeriodMs)
    out.warnings.push_back("delta exceeds one frame period (11.11 ms) -- suspect the IMU "
                           "timestamp pipeline.");
  if (out.biasDeg.norm() > 3.0)
    out.warnings.push_back("gyro bias magnitude > 3 deg/s -- suspect gyro units/scale or no "
                           "static segment.");
  if (!staticBiasFound && !m_config.refineBias)
    out.warnings.push_back("no static segment found and bias not refined -- gyro bias is ZERO. "
                           "Capture a few seconds at rest, or pass --refine-bias.");
  if (out.medianInlierRatio < m_config.inlierRatioThreshold)
    out.warnings.push_back("Stage-1 median inlier ratio is low -- excess parallax/translation "
                           "or dynamic scene; check the capture.");
  if (out.stage2CorrelationPeak < 0.5)
    out.warnings.push_back("Stage-2 correlation peak is weak -- insufficient or degenerate "
                           "rotational excitation.");
  if (out.residualRmsDeg > 0.3)
    out.warnings.push_back("Residual RMS > 0.3 deg after convergence -- investigate before "
                           "trusting the result.");

  // Observability: warn if the used rotation directions are nearly single-axis.
  {
    Eigen::Matrix3d dirCov = Eigen::Matrix3d::Zero();
    for (const PairMeasurement& m : m_pairs)
      if (m.usableForRotation)
        dirCov += m.thetaCam * m.thetaCam.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(dirCov);
    const double minEv = es.eigenvalues()(0), maxEv = es.eigenvalues()(2);
    if (maxEv > 0.0 && (minEv / maxEv) < 1e-3)
      out.warnings.push_back("Rotation excitation is nearly single-axis -- two of three axes "
                             "are poorly constrained. Excite pan, tilt, and roll.");
  }

  return true;
}

} // namespace CameraImuCalib
