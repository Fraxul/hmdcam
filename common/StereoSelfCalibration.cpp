#include "common/StereoSelfCalibration.h"
#include "common/AsyncGpuDumpRing.h"
#include "common/CameraSystem.h"
#include "common/DepthMapGenerator.h"
#include "common/ICameraProvider.h"
#include "common/IMUService.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/cuda/RHICUDA.h"
#include "imgui.h"
#include <assert.h>
#include <stdio.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow" // Ceres headers generate some -Wshadow warnings, just ignore them.
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#pragma clang diagnostic pop
#include <cuda.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

// Snapshots are sporadic (periodic solves, or manual debug captures), so the
// ring only needs enough slots to cover every stereo view being captured in
// the same frame plus one in-flight write.
constexpr size_t kSnapshotRingSlots = 4;

// Temporal voting: a correction is only auto-applied after this many
// consecutive gates-passed solves agree within kVoteAgreementDegrees.
// Pathological scenes can produce a plausible-but-wrong solve, but they don't
// produce a *consistent* wrong answer across separate captures.
constexpr uint32_t kVotesRequired = 3;
constexpr float kVoteAgreementDegrees = 0.05f;
// Skip the auto-apply when the agreed correction is this close to the current
// target -- nothing worth updating (also throttles sidecar rewrites).
constexpr float kMinAutoApplyDeltaDegrees = 0.005f;
// Consecutive quiescent-IMU frames required before an automatic capture.
constexpr uint32_t kStableFramesRequired = 30;

constexpr const char* kOnlineCalibrationFilename = "onlineCalibration.yml";

// atan2 formulation: precise for the tiny angles the vote comparison cares
// about, where acos(dot) loses everything to float cancellation near 1.
static float angleBetweenDegrees(const glm::quat& a, const glm::quat& b) {
  const glm::quat delta = glm::inverse(a) * b;
  return glm::degrees(2.0f * std::atan2(glm::length(glm::vec3(delta.x, delta.y, delta.z)), std::abs(delta.w)));
}

StereoSelfCalibration::StereoSelfCalibration(CameraSystem* cameraSystem, DepthMapGenerator* depthMapGenerator) :
  m_cameraSystem(cameraSystem),
  m_depthMapGenerator(depthMapGenerator) {

  m_voteState.resize(cameraSystem->views());
  m_lastAutoCaptureTime = std::chrono::steady_clock::now();
  loadSavedCorrections();
}

StereoSelfCalibration::~StereoSelfCalibration() {
  if (m_dumpRing) {
    m_dumpRing->drainBlocking();
    delete m_dumpRing;
  }
}

void StereoSelfCalibration::processFrame(IMUFrame* imuFrame) {
  if (isIMUStable(imuFrame))
    m_consecutiveStableFrames += 1;
  else
    m_consecutiveStableFrames = 0;

  // Manual captures fire unconditionally; automatic captures wait for the
  // interval to elapse and the rig to have been still long enough that the
  // snapshot isn't contaminated by motion blur or residual RS distortion.
  bool captureNow = m_captureRequested;
  m_captureRequested = false;
  if (!captureNow && m_autoCalibrationEnabled && m_consecutiveStableFrames >= kStableFramesRequired) {
    const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
    if (std::chrono::duration<float>(now - m_lastAutoCaptureTime).count() >= m_autoIntervalSeconds)
      captureNow = true;
  }

  if (captureNow) {
    m_lastAutoCaptureTime = std::chrono::steady_clock::now();
    for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
      if (!m_cameraSystem->viewAtIndex(viewIdx).isStereo)
        continue;
      if (!captureView(viewIdx))
        m_debugDroppedSnapshots += 1;
    }
  }

  consumeSolveResults();
}

bool StereoSelfCalibration::isIMUStable(IMUFrame* imuFrame) const {
  if (imuFrame == nullptr || imuFrame->validSampleCount == 0)
    return true; // No IMU data: don't block automatic calibration on it.
  for (uint32_t sampleIdx = 0; sampleIdx < imuFrame->validSampleCount; ++sampleIdx) {
    if (glm::length(imuFrame->samples[sampleIdx].gyroDPS) > m_gyroStabilityThresholdDPS)
      return false;
  }
  return true;
}

void StereoSelfCalibration::consumeSolveResults() {
  std::lock_guard<std::mutex> guard(m_matchStatsLock);
  for (size_t viewIdx = 0; viewIdx < m_matchStats.size() && viewIdx < m_voteState.size(); ++viewIdx) {
    const ViewMatchStats& stats = m_matchStats[viewIdx];
    VoteState& voteState = m_voteState[viewIdx];
    if (!stats.solve.attempted)
      continue;
    if (static_cast<int64_t>(stats.snapshot.sequence) == voteState.lastConsumedSequence)
      continue; // already voted on this solve
    voteState.lastConsumedSequence = stats.snapshot.sequence;

    if (!stats.solve.gatesPassed)
      continue; // Scene-quality failure: keep the votes we have.
    if (stats.snapshot.calibrationRevision != m_cameraSystem->calibrationDataRevision()) {
      voteState.votes.clear(); // Solved against a stale base calibration.
      continue;
    }

    const glm::quat& candidate = stats.solve.correctionCandidate;
    for (const glm::quat& vote : voteState.votes) {
      if (angleBetweenDegrees(candidate, vote) > kVoteAgreementDegrees) {
        voteState.votes.clear(); // Disagreement restarts the run...
        break;
      }
    }
    voteState.votes.push_back(candidate); // ...with the candidate as its first vote.
    if (voteState.votes.size() < kVotesRequired)
      continue;

    // Hemisphere-aligned average of the agreeing votes.
    glm::vec4 accumulator(0.0f);
    for (const glm::quat& vote : voteState.votes) {
      const float sign = (glm::dot(vote, voteState.votes[0]) < 0.0f) ? -1.0f : 1.0f;
      accumulator += sign * glm::vec4(vote.x, vote.y, vote.z, vote.w);
    }
    const glm::quat average = glm::normalize(glm::quat(accumulator.w, accumulator.x, accumulator.y, accumulator.z));
    voteState.votes.clear();

    if (!m_autoCalibrationEnabled)
      continue; // Manual mode: votes are informational only.
    if (angleBetweenDegrees(average, m_cameraSystem->viewAtIndex(viewIdx).stereoCorrectionTarget) < kMinAutoApplyDeltaDegrees)
      continue; // Already aligned; nothing worth updating.
    applyCorrection(viewIdx, average);
  }
}

void StereoSelfCalibration::applyCorrection(size_t viewIdx, const glm::quat& correction) {
  const float deltaDegrees = angleBetweenDegrees(m_cameraSystem->viewAtIndex(viewIdx).stereoCorrectionTarget, correction);
  m_cameraSystem->viewAtIndex(viewIdx).stereoCorrectionTarget = correction;
  printf("StereoSelfCalibration: applied correction for view %zu: quat [w=%f x=%f y=%f z=%f] (%.4f deg change)\n",
    viewIdx, correction.w, correction.x, correction.y, correction.z, deltaDegrees);
  saveCorrections();
}

void StereoSelfCalibration::loadSavedCorrections() {
  cv::FileStorage fileStorage;
  try {
    if (!fileStorage.open(kOnlineCalibrationFilename, cv::FileStorage::READ))
      return; // No saved corrections yet.
  } catch (const cv::Exception& ex) {
    printf("StereoSelfCalibration: unable to read %s: %s\n", kOnlineCalibrationFilename, ex.what());
    return;
  }

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    CameraSystem::View& v = m_cameraSystem->viewAtIndex(viewIdx);
    if (!v.isStereo)
      continue;
    char key[32];
    snprintf(key, sizeof(key), "view%zuCorrection", viewIdx);
    cv::FileNode node = fileStorage[key];
    if (node.empty())
      continue;
    cv::Vec4d wxyz;
    node >> wxyz;
    glm::quat correction(static_cast<float>(wxyz[0]), static_cast<float>(wxyz[1]), static_cast<float>(wxyz[2]), static_cast<float>(wxyz[3]));
    const float norm = glm::length(correction);
    if (!(std::abs(norm - 1.0f) < 0.01f)) { // also rejects NaN
      printf("StereoSelfCalibration: ignoring malformed saved correction for view %zu\n", viewIdx);
      continue;
    }
    correction = correction / norm;
    // Snap both active and target at startup -- there's nothing to blend from.
    v.stereoCorrectionActive = correction;
    v.stereoCorrectionTarget = correction;
    printf("StereoSelfCalibration: loaded correction for view %zu: quat [w=%f x=%f y=%f z=%f]\n",
      viewIdx, correction.w, correction.x, correction.y, correction.z);
  }
}

// Small file, infrequent writes (only on apply): synchronous on the render
// thread, which also keeps concurrent writers impossible.
void StereoSelfCalibration::saveCorrections() const {
  cv::FileStorage fileStorage(kOnlineCalibrationFilename, cv::FileStorage::WRITE);
  if (!fileStorage.isOpened()) {
    printf("StereoSelfCalibration: unable to write %s\n", kOnlineCalibrationFilename);
    return;
  }
  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    const CameraSystem::View& v = m_cameraSystem->viewAtIndex(viewIdx);
    if (!v.isStereo)
      continue;
    char key[32];
    snprintf(key, sizeof(key), "view%zuCorrection", viewIdx);
    const glm::quat& q = v.stereoCorrectionTarget;
    fileStorage << key << cv::Vec4d(q.w, q.x, q.y, q.z);
  }
}

// Issues a tightly-packed DtoH copy of a pitched device mat into a slot host region.
static void copyDeviceMatToHost(void* dstHost, const cv::cuda::GpuMat& src, size_t rowBytes, size_t height, CUstream stream) {
  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(copyDescriptor));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.srcDevice = (CUdeviceptr) src.cudaPtr();
  copyDescriptor.srcPitch = src.step;
  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
  copyDescriptor.dstHost = dstHost;
  copyDescriptor.dstPitch = rowBytes; // tightly-packed host destination
  copyDescriptor.WidthInBytes = rowBytes;
  copyDescriptor.Height = height;
  CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
}

bool StereoSelfCalibration::captureView(size_t viewIdx) {
  cv::cuda::GpuMat leftLuma = m_depthMapGenerator->rectifiedLumaGpuMat(viewIdx, 0);
  cv::cuda::GpuMat rightLuma = m_depthMapGenerator->rectifiedLumaGpuMat(viewIdx, 1);
  cv::cuda::GpuMat disparity = m_depthMapGenerator->currentDisparityGpuMat(viewIdx);
  if (leftLuma.empty() || rightLuma.empty() || disparity.empty())
    return false; // Backend hasn't populated this view (yet).

  assert(leftLuma.type() == CV_8U && rightLuma.type() == CV_8U);
  assert(leftLuma.size() == rightLuma.size());

  SnapshotMetadata metadata;
  metadata.viewIdx = viewIdx;
  metadata.sequence = m_captureSequence++;
  metadata.lumaWidth = leftLuma.cols;
  metadata.lumaHeight = leftLuma.rows;
  metadata.disparityWidth = disparity.cols;
  metadata.disparityHeight = disparity.rows;
  metadata.disparityCvType = disparity.type();
  metadata.disparityPrescale = m_depthMapGenerator->disparityPrescale();
  metadata.maxDisparityRaw = m_depthMapGenerator->maxDisparityRaw();

  const CameraSystem::View& v = m_cameraSystem->viewAtIndex(viewIdx);
  metadata.correctionAtCapture = v.stereoCorrectionActive;
  metadata.calibrationRevision = m_cameraSystem->calibrationDataRevision();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      metadata.stereoProjectionRight[(r * 3) + c] = v.stereoProjection[1].at<double>(r, c);
      metadata.stereoRectificationRight[(r * 3) + c] = v.stereoRectification[1].at<double>(r, c);
    }
  }
  metadata.streamWidth = m_cameraSystem->cameraProvider()->streamWidth();
  metadata.streamHeight = m_cameraSystem->cameraProvider()->streamHeight();

  const size_t lumaBytes = static_cast<size_t>(metadata.lumaWidth) * metadata.lumaHeight;
  const size_t disparityRowBytes = static_cast<size_t>(metadata.disparityWidth) * disparity.elemSize();
  metadata.lumaOffset[0] = 0;
  metadata.lumaOffset[1] = lumaBytes;
  metadata.disparityOffset = 2 * lumaBytes;
  const size_t totalBytes = (2 * lumaBytes) + (disparityRowBytes * metadata.disparityHeight);

  if (!m_dumpRing)
    m_dumpRing = new AsyncGpuDumpRing(kSnapshotRingSlots, totalBytes);

  AsyncGpuDumpRing::Slot* slot = m_dumpRing->acquire();
  if (!slot)
    return false; // Pool exhausted -- drop this snapshot.
  assert(slot->capacity >= totalBytes); // Stream dimensions are fixed for the process lifetime.

  // The depth backend produced this frame's rectified luma and disparity on
  // the default async stream, so issuing the copies there orders them after
  // those writes (and before next frame's).
  uint8_t* base = static_cast<uint8_t*>(slot->hostPtr);
  copyDeviceMatToHost(base + metadata.lumaOffset[0], leftLuma, metadata.lumaWidth, metadata.lumaHeight, RHICUDA::defaultAsyncStream);
  copyDeviceMatToHost(base + metadata.lumaOffset[1], rightLuma, metadata.lumaWidth, metadata.lumaHeight, RHICUDA::defaultAsyncStream);
  copyDeviceMatToHost(base + metadata.disparityOffset, disparity, disparityRowBytes, metadata.disparityHeight, RHICUDA::defaultAsyncStream);
  CUDA_CHECK(cuEventRecord(slot->copyDoneEvent, RHICUDA::defaultAsyncStream));

  const bool dumpPNGs = m_debugDumpPNGs;
  m_dumpRing->dispatch(slot, [this, metadata, dumpPNGs](AsyncGpuDumpRing::Slot* writeSlot) {
    processSnapshot(metadata, static_cast<const uint8_t*>(writeSlot->hostPtr), dumpPNGs);
    m_debugCompletedSnapshots += 1;
  });
  return true;
}

// Ring worker thread: run the feature-match pass over a completed snapshot,
// publish the result for the UI / solve stage, and optionally dump PNGs.
void StereoSelfCalibration::processSnapshot(const SnapshotMetadata& metadata, const uint8_t* slotBase, bool dumpPNGs) {
  cv::Mat leftLuma(metadata.lumaHeight, metadata.lumaWidth, CV_8U, const_cast<uint8_t*>(slotBase + metadata.lumaOffset[0]));
  cv::Mat rightLuma(metadata.lumaHeight, metadata.lumaWidth, CV_8U, const_cast<uint8_t*>(slotBase + metadata.lumaOffset[1]));
  cv::Mat rawDisparity(metadata.disparityHeight, metadata.disparityWidth, metadata.disparityCvType, const_cast<uint8_t*>(slotBase + metadata.disparityOffset));

  // Raw subpixel units -> float pixel units, once, up front.
  cv::Mat disparityPixels;
  rawDisparity.convertTo(disparityPixels, CV_32F, metadata.disparityPrescale);

  ViewMatchStats stats;
  matchFeatures(metadata, leftLuma, rightLuma, disparityPixels, stats);
  solveCorrection(stats);

  printf("StereoSelfCalibration: snapshot %04u view %zu: %u detected, %u seeded, %u matched, %u inliers; "
         "deltaY mean %+.3f px RMS %.3f px, coverage %u/%u, %.1f ms\n",
    metadata.sequence, metadata.viewIdx,
    stats.detectedCount, stats.seededCount, stats.matchedCount, stats.inlierCount,
    stats.meanDeltaY, stats.rmsDeltaY, stats.coverageCells, kCoverageGridSize * kCoverageGridSize,
    stats.matchMilliseconds);
  if (stats.solve.attempted) {
    printf("StereoSelfCalibration: solve %04u view %zu: G=[%+.4f %+.4f %+.4f] deg, RMS %.3f -> %.3f px, "
           "inlier fraction %.2f, %s, %.1f ms\n",
      metadata.sequence, metadata.viewIdx,
      glm::degrees(stats.solve.rectRotation.x), glm::degrees(stats.solve.rectRotation.y), glm::degrees(stats.solve.rectRotation.z),
      stats.solve.rmsBeforePixels, stats.solve.rmsAfterPixels, stats.solve.solveInlierFraction,
      stats.solve.gatesPassed ? "gates PASS" : stats.solve.gateFailure, stats.solve.solveMilliseconds);
  } else if (stats.solve.gateFailure[0]) {
    printf("StereoSelfCalibration: solve %04u view %zu: skipped (%s)\n", metadata.sequence, metadata.viewIdx, stats.solve.gateFailure);
  }

  if (dumpPNGs) {
    debugDumpSnapshot(metadata, slotBase);
    debugDumpMatchAnnotations(leftLuma, stats);
  }

  {
    std::lock_guard<std::mutex> guard(m_matchStatsLock);
    if (m_matchStats.size() <= metadata.viewIdx)
      m_matchStats.resize(metadata.viewIdx + 1);
    m_matchStats[metadata.viewIdx] = std::move(stats);
  }
}

// Detect corners in the left rectified image and match them into the right
// image with forward+backward pyramidal LK. The horizontal search is seeded
// from the SGM disparity map (x_right = x_left - disparity), so LK only has to
// refine to subpixel rather than search the whole disparity range.
void StereoSelfCalibration::matchFeatures(const SnapshotMetadata& metadata, const cv::Mat& leftLuma, const cv::Mat& rightLuma, const cv::Mat& disparityPixels, ViewMatchStats& result) const {
  const int64_t startTick = cv::getTickCount();

  constexpr int kMaxCorners = 400;
  constexpr double kCornerQualityLevel = 0.01;
  constexpr float kBorderMarginPixels = 12.0f; // keep LK windows inside the image
  // Inlier prefilter on vertical residual. Sized to match the solve's 0.5deg
  // magnitude gate (~8 px at a ~900 px focal length) -- a tighter clip
  // discards the strongest-signal periphery matches under real drift and
  // biases the solve toward underestimation.
  constexpr float kMaxDeltaYPixels = 8.0f;
  constexpr float kMaxBacktrackErrorPixels = 0.5f;
  const cv::Size lkWindowSize(21, 21);
  constexpr int kLkPyramidLevels = 3;

  result.snapshot = metadata;

  // Shi-Tomasi corners on the left image. minDistance enforces spatial spread
  // so a single high-texture patch can't dominate the residual field.
  std::vector<cv::Point2f> corners;
  const double minCornerDistance = std::max(8.0, static_cast<double>(leftLuma.cols) / 60.0);
  cv::goodFeaturesToTrack(leftLuma, corners, kMaxCorners, kCornerQualityLevel, minCornerDistance);
  result.detectedCount = corners.size();

  // Seed each corner's right-image position from the disparity map. Disparity
  // is measured at the disparity map's own resolution; scale offsets up to
  // luma resolution when they differ. Corners without a usable disparity
  // (invalid/out-of-range, typically low-confidence regions) are skipped.
  const float lumaFromDisparityScale = static_cast<float>(leftLuma.cols) / static_cast<float>(disparityPixels.cols);
  const float maxDisparityPixels = static_cast<float>(metadata.maxDisparityRaw) * metadata.disparityPrescale;
  std::vector<cv::Point2f> leftPoints, rightPoints;
  leftPoints.reserve(corners.size());
  rightPoints.reserve(corners.size());
  for (const cv::Point2f& corner : corners) {
    if (corner.x < kBorderMarginPixels || corner.y < kBorderMarginPixels ||
      corner.x >= (leftLuma.cols - kBorderMarginPixels) || corner.y >= (leftLuma.rows - kBorderMarginPixels))
      continue;

    const int disparityX = std::min(static_cast<int>(corner.x / lumaFromDisparityScale), disparityPixels.cols - 1);
    const int disparityY = std::min(static_cast<int>(corner.y / lumaFromDisparityScale), disparityPixels.rows - 1);
    const float disparity = disparityPixels.at<float>(disparityY, disparityX);
    if (!(disparity > 0.5f) || disparity > maxDisparityPixels)
      continue; // invalid, missing, or out-of-range (also rejects NaN)

    const cv::Point2f rightGuess(corner.x - (disparity * lumaFromDisparityScale), corner.y);
    if (rightGuess.x < kBorderMarginPixels)
      continue;

    leftPoints.push_back(corner);
    rightPoints.push_back(rightGuess);
  }
  result.seededCount = leftPoints.size();
  if (leftPoints.empty()) {
    result.matchMilliseconds = static_cast<float>((cv::getTickCount() - startTick) * 1000.0 / cv::getTickFrequency());
    return;
  }

  // Forward LK refines the seeded guesses to subpixel; backward LK re-tracks
  // the results to the left image, and matches that don't round-trip are
  // rejected. This is the standard forward-backward consistency filter.
  const cv::TermCriteria lkCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
  std::vector<uint8_t> forwardStatus, backwardStatus;
  std::vector<float> forwardError, backwardError;
  cv::calcOpticalFlowPyrLK(leftLuma, rightLuma, leftPoints, rightPoints, forwardStatus, forwardError,
    lkWindowSize, kLkPyramidLevels, lkCriteria, cv::OPTFLOW_USE_INITIAL_FLOW);

  std::vector<cv::Point2f> backtrackPoints = leftPoints;
  cv::calcOpticalFlowPyrLK(rightLuma, leftLuma, rightPoints, backtrackPoints, backwardStatus, backwardError,
    lkWindowSize, kLkPyramidLevels, lkCriteria, cv::OPTFLOW_USE_INITIAL_FLOW);

  double deltaYSum = 0.0, deltaYSquaredSum = 0.0;
  uint32_t coverageMask = 0;
  result.leftPoints.reserve(leftPoints.size());
  result.rightPoints.reserve(leftPoints.size());
  for (size_t i = 0; i < leftPoints.size(); ++i) {
    if (!forwardStatus[i] || !backwardStatus[i])
      continue;
    const cv::Point2f backtrackError = backtrackPoints[i] - leftPoints[i];
    if ((backtrackError.x * backtrackError.x + backtrackError.y * backtrackError.y) >
      (kMaxBacktrackErrorPixels * kMaxBacktrackErrorPixels))
      continue;
    result.matchedCount += 1;

    const float deltaY = rightPoints[i].y - leftPoints[i].y;
    if (std::abs(deltaY) > kMaxDeltaYPixels)
      continue;

    result.leftPoints.push_back(leftPoints[i]);
    result.rightPoints.push_back(rightPoints[i]);
    deltaYSum += deltaY;
    deltaYSquaredSum += static_cast<double>(deltaY) * deltaY;

    const uint32_t cellX = std::min(static_cast<uint32_t>((leftPoints[i].x * kCoverageGridSize) / leftLuma.cols), kCoverageGridSize - 1);
    const uint32_t cellY = std::min(static_cast<uint32_t>((leftPoints[i].y * kCoverageGridSize) / leftLuma.rows), kCoverageGridSize - 1);
    coverageMask |= 1u << ((cellY * kCoverageGridSize) + cellX);
  }

  result.inlierCount = result.leftPoints.size();
  if (result.inlierCount > 0) {
    result.valid = true;
    result.meanDeltaY = static_cast<float>(deltaYSum / result.inlierCount);
    result.rmsDeltaY = static_cast<float>(std::sqrt(deltaYSquaredSum / result.inlierCount));
  }
  result.coverageMask = coverageMask;
  result.coverageCells = __builtin_popcount(coverageMask);
  result.matchMilliseconds = static_cast<float>((cv::getTickCount() - startTick) * 1000.0 / cv::getTickFrequency());
}

// The vertical residual barely observes yaw (rotation about the rectified y
// axis): it only enters through second-order x*y perspective terms, while its
// first-order effect is horizontal disparity -- i.e. depth-scale bias. This
// prior pins it near zero unless the data overwhelmingly disagrees.
constexpr double kYawPriorSigmaRadians = 0.02 * M_PI / 180.0;

namespace {

// One matched pair's vertical alignment error under a candidate rectified-
// frame rotation G (angle-axis g). The corrected position of a right-image
// point u is H*u with H = A * R(g) * A^-1, A = P2 at stream resolution;
// rayRight = A^-1 * [x_right, y_right, 1] is precomputed. The residual is the
// predicted right row minus the left row it should align with, in stream-res
// pixels.
struct VerticalDisparityResidual {
  VerticalDisparityResidual(const double* a, const double* rayRight, double yLeft) :
    m_yLeft(yLeft) {
    memcpy(m_a, a, sizeof(m_a));
    memcpy(m_ray, rayRight, sizeof(m_ray));
  }

  template <typename T>
  bool operator()(const T* g, T* residual) const {
    const T ray[3] = {T(m_ray[0]), T(m_ray[1]), T(m_ray[2])};
    T rotated[3];
    ceres::AngleAxisRotatePoint(g, ray, rotated);
    const T predictedY = (T(m_a[3]) * rotated[0]) + (T(m_a[4]) * rotated[1]) + (T(m_a[5]) * rotated[2]);
    const T predictedW = (T(m_a[6]) * rotated[0]) + (T(m_a[7]) * rotated[1]) + (T(m_a[8]) * rotated[2]);
    residual[0] = (predictedY / predictedW) - T(m_yLeft);
    return true;
  }

  double m_a[9];
  double m_ray[3];
  double m_yLeft;
};

struct YawPriorResidual {
  template <typename T>
  bool operator()(const T* g, T* residual) const {
    residual[0] = g[1] / T(kYawPriorSigmaRadians);
    return true;
  }
};

// glm::dmat3 (column-major) from a row-major double[9].
glm::dmat3 dmat3FromRowMajor(const double* a) {
  glm::dmat3 m;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      m[c][r] = a[(r * 3) + c];
    }
  }
  return m;
}

} // anonymous namespace

// Solve for the small rectified-frame rotation G that zeroes the vertical
// residuals of the match inliers, then compose it into a full replacement
// value for View::stereoCorrectionTarget:
//
//   C_new = C_capture * R2^T * G^T * R2
//
// (R2 conjugates the rect-frame rotation back into the right camera's frame,
// where the correction is applied to rsIRBase; see stereoCorrection* in
// CameraSystem.h.) Match-quality and solution-quality gates guard against a
// low-information or degenerate scene producing a plausible-but-wrong answer.
void StereoSelfCalibration::solveCorrection(ViewMatchStats& stats) const {
  SolveResult& result = stats.solve;
  const SnapshotMetadata& metadata = stats.snapshot;
  const int64_t startTick = cv::getTickCount();

  constexpr uint32_t kMinInliers = 60;
  constexpr uint32_t kMinCoverageRows = 3, kMinCoverageColumns = 3;
  constexpr double kHuberLossScalePixels = 2.0; // stage 1: wide basin for the systematic offset
  constexpr double kCauchyLossScalePixels = 0.5; // stage 2: tight outlier rejection near the solution
  constexpr float kMaxCandidateAngleDegrees = 0.5f;
  constexpr float kMaxRmsAfterPixels = 0.9f;
  constexpr double kSolveInlierThresholdPixels = 0.75;
  constexpr float kMinSolveInlierFraction = 0.7f;

  // ----- Match-quality gates (before spending any solver time) -----
  if (stats.inlierCount < kMinInliers) {
    snprintf(result.gateFailure, sizeof(result.gateFailure), "inliers %u < %u", stats.inlierCount, kMinInliers);
    return;
  }
  // Coverage spread: roll needs x-spread, y-scale needs y-spread. Require
  // occupied cells in at least kMinCoverage rows AND columns of the grid.
  uint32_t rowMask = 0, columnMask = 0;
  for (uint32_t cell = 0; cell < kCoverageGridSize * kCoverageGridSize; ++cell) {
    if (stats.coverageMask & (1u << cell)) {
      rowMask |= 1u << (cell / kCoverageGridSize);
      columnMask |= 1u << (cell % kCoverageGridSize);
    }
  }
  if (__builtin_popcount(rowMask) < static_cast<int>(kMinCoverageRows) ||
    __builtin_popcount(columnMask) < static_cast<int>(kMinCoverageColumns)) {
    snprintf(result.gateFailure, sizeof(result.gateFailure), "coverage %ur x %uc", __builtin_popcount(rowMask), __builtin_popcount(columnMask));
    return;
  }

  result.attempted = true;

  // Match coordinates are luma-resolution; the projection is stream-
  // resolution. Standard pixel-center resampling: x_stream = (x + 0.5)*k - 0.5.
  const double scaleX = static_cast<double>(metadata.streamWidth) / metadata.lumaWidth;
  const double scaleY = static_cast<double>(metadata.streamHeight) / metadata.lumaHeight;

  const cv::Matx33d a(metadata.stereoProjectionRight); // row-major, same layout as Matx
  const cv::Matx33d aInverse = a.inv();

  struct ResidualData {
    double ray[3];
    double yLeft;
  };
  std::vector<ResidualData> residualData(stats.inlierCount);

  double sumSquaredBefore = 0.0;
  for (size_t i = 0; i < stats.inlierCount; ++i) {
    const double rightX = ((stats.rightPoints[i].x + 0.5) * scaleX) - 0.5;
    const double rightY = ((stats.rightPoints[i].y + 0.5) * scaleY) - 0.5;
    const double leftY = ((stats.leftPoints[i].y + 0.5) * scaleY) - 0.5;

    const cv::Vec3d ray = aInverse * cv::Vec3d(rightX, rightY, 1.0);
    residualData[i].ray[0] = ray[0];
    residualData[i].ray[1] = ray[1];
    residualData[i].ray[2] = ray[2];
    residualData[i].yLeft = leftY;

    sumSquaredBefore += (rightY - leftY) * (rightY - leftY);
  }
  result.rmsBeforePixels = static_cast<float>(std::sqrt(sumSquaredBefore / stats.inlierCount));

  // Two-stage robust solve. Right after drift (or an injected error) every
  // residual starts several pixels from zero; a tight Cauchy loss there
  // downweights the true signal as if it were outliers and underestimates the
  // correction. Stage 1 uses a wide Huber to absorb the systematic offset;
  // stage 2 re-solves warm-started with the tight Cauchy so genuine outliers
  // are rejected once the bulk of the residuals sit near zero.
  double g[3] = {0.0, 0.0, 0.0};
  for (int stage = 0; stage < 2; ++stage) {
    ceres::Problem problem;
    for (size_t i = 0; i < stats.inlierCount; ++i) {
      ceres::LossFunction* loss = (stage == 0)
        ? static_cast<ceres::LossFunction*>(new ceres::HuberLoss(kHuberLossScalePixels))
        : static_cast<ceres::LossFunction*>(new ceres::CauchyLoss(kCauchyLossScalePixels));
      problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<VerticalDisparityResidual, 1, 3>(
          new VerticalDisparityResidual(a.val, residualData[i].ray, residualData[i].yLeft)),
        loss, g);
    }
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<YawPriorResidual, 1, 3>(new YawPriorResidual()), nullptr, g);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = 50;
    options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    result.converged = (summary.termination_type == ceres::CONVERGENCE);
  }
  result.rectRotation = glm::vec3(g[0], g[1], g[2]);

  // Post-solve residual evaluation (plain double math, no robust loss).
  double rotation[9]; // column-major: element (r, c) = rotation[c*3 + r]
  ceres::AngleAxisToRotationMatrix(g, rotation);
  double sumSquaredAfter = 0.0;
  uint32_t solveInlierCount = 0;
  for (const ResidualData& data : residualData) {
    double rotated[3];
    for (int r = 0; r < 3; ++r) {
      rotated[r] = (rotation[r] * data.ray[0]) + (rotation[3 + r] * data.ray[1]) + (rotation[6 + r] * data.ray[2]);
    }
    const double predictedY = (a.val[3] * rotated[0]) + (a.val[4] * rotated[1]) + (a.val[5] * rotated[2]);
    const double predictedW = (a.val[6] * rotated[0]) + (a.val[7] * rotated[1]) + (a.val[8] * rotated[2]);
    const double residual = (predictedY / predictedW) - data.yLeft;
    sumSquaredAfter += residual * residual;
    if (std::abs(residual) < kSolveInlierThresholdPixels)
      solveInlierCount += 1;
  }
  result.rmsAfterPixels = static_cast<float>(std::sqrt(sumSquaredAfter / stats.inlierCount));
  result.solveInlierFraction = static_cast<float>(solveInlierCount) / stats.inlierCount;

  // Compose the full candidate correction: C_new = C_cap * R2^T * G^T * R2.
  const double solvedAngle = std::sqrt((g[0] * g[0]) + (g[1] * g[1]) + (g[2] * g[2]));
  glm::dquat qG(1.0, 0.0, 0.0, 0.0);
  if (solvedAngle > 0.0)
    qG = glm::angleAxis(solvedAngle, glm::dvec3(g[0], g[1], g[2]) / solvedAngle);
  const glm::dquat qR2 = glm::quat_cast(dmat3FromRowMajor(metadata.stereoRectificationRight));
  const glm::dquat qCapture(metadata.correctionAtCapture);
  result.correctionCandidate = glm::quat(glm::normalize(qCapture * glm::conjugate(qR2) * glm::conjugate(qG) * qR2));

  // ----- Solution-quality gates -----
  const float solvedAngleDegrees = glm::degrees(static_cast<float>(solvedAngle));
  if (!result.converged)
    snprintf(result.gateFailure, sizeof(result.gateFailure), "no convergence");
  else if (solvedAngleDegrees > kMaxCandidateAngleDegrees)
    snprintf(result.gateFailure, sizeof(result.gateFailure), "magnitude %.3f deg", solvedAngleDegrees);
  else if (result.rmsAfterPixels > kMaxRmsAfterPixels)
    snprintf(result.gateFailure, sizeof(result.gateFailure), "RMS after %.3f px", result.rmsAfterPixels);
  else if (result.solveInlierFraction < kMinSolveInlierFraction)
    snprintf(result.gateFailure, sizeof(result.gateFailure), "inlier frac %.2f", result.solveInlierFraction);
  else
    result.gatesPassed = true;

  result.solveMilliseconds = static_cast<float>((cv::getTickCount() - startTick) * 1000.0 / cv::getTickFrequency());
}

// Ring worker thread: serialize a completed snapshot to PNGs for visual
// verification. The luma planes are written as-is; the disparity plane is
// normalized to an 8-bit preview (0 .. maxDisparityRaw -> 0 .. 255).
void StereoSelfCalibration::debugDumpSnapshot(const SnapshotMetadata& metadata, const uint8_t* slotBase) {
  char filename[128];
  const char* eyeName[2] = {"left", "right"};
  for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
    cv::Mat luma(metadata.lumaHeight, metadata.lumaWidth, CV_8U, const_cast<uint8_t*>(slotBase + metadata.lumaOffset[eyeIdx]));
    snprintf(filename, sizeof(filename), "selfcal-%04u-view%zu-%s.png", metadata.sequence, metadata.viewIdx, eyeName[eyeIdx]);
    cv::imwrite(filename, luma);
  }

  cv::Mat rawDisparity(metadata.disparityHeight, metadata.disparityWidth, metadata.disparityCvType, const_cast<uint8_t*>(slotBase + metadata.disparityOffset));
  cv::Mat disparityPreview;
  rawDisparity.convertTo(disparityPreview, CV_8U, 255.0 / static_cast<double>(metadata.maxDisparityRaw));
  snprintf(filename, sizeof(filename), "selfcal-%04u-view%zu-disparity.png", metadata.sequence, metadata.viewIdx);
  cv::imwrite(filename, disparityPreview);
}

// Ring worker thread: write the left luma image annotated with the inlier
// matches. Each inlier gets a circle plus a 20x-exaggerated vertical whisker
// showing its deltaY residual, colored by sign and magnitude (green ~0,
// red positive, blue negative, saturating at +/-2 px). The spatial pattern is
// diagnostic: uniform color = pitch error, left/right gradient = roll,
// top/bottom gradient = y-scale.
void StereoSelfCalibration::debugDumpMatchAnnotations(const cv::Mat& leftLuma, const ViewMatchStats& stats) {
  cv::Mat canvas;
  cv::cvtColor(leftLuma, canvas, cv::COLOR_GRAY2BGR);

  for (size_t i = 0; i < stats.leftPoints.size(); ++i) {
    const float deltaY = stats.rightPoints[i].y - stats.leftPoints[i].y;
    const float t = std::max(-1.0f, std::min(1.0f, deltaY / 2.0f));
    const cv::Scalar color = (t >= 0.0f)
      ? cv::Scalar(0.0, 255.0 * (1.0f - t), 255.0 * t) // green -> red
      : cv::Scalar(255.0 * -t, 255.0 * (1.0f + t), 0.0); // green -> blue

    const cv::Point center(cvRound(stats.leftPoints[i].x), cvRound(stats.leftPoints[i].y));
    cv::circle(canvas, center, 4, color, 1);
    cv::line(canvas, center, center + cv::Point(0, cvRound(deltaY * 20.0f)), color, 1);
  }

  char text[128];
  snprintf(text, sizeof(text), "inliers %u/%u  dY mean %+.3f  rms %.3f  coverage %u/%u",
    stats.inlierCount, stats.detectedCount, stats.meanDeltaY, stats.rmsDeltaY,
    stats.coverageCells, kCoverageGridSize * kCoverageGridSize);
  cv::putText(canvas, text, cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);

  char filename[128];
  snprintf(filename, sizeof(filename), "selfcal-%04u-view%zu-matches.png", stats.snapshot.sequence, stats.snapshot.viewIdx);
  cv::imwrite(filename, canvas);
}

void StereoSelfCalibration::renderIMGUI() {
  ImGui::Text("Online Self-Calibration");
  ImGui::Checkbox("Auto-calibrate", &m_autoCalibrationEnabled);
  if (m_autoCalibrationEnabled) {
    ImGui::SliderFloat("Interval", &m_autoIntervalSeconds, 1.0f, 60.0f, "%.0f s");
    ImGui::SliderFloat("Gyro stability", &m_gyroStabilityThresholdDPS, 0.5f, 10.0f, "%.1f deg/s");
    ImGui::Text("IMU stable frames: %u/%u", std::min(m_consecutiveStableFrames, kStableFramesRequired), kStableFramesRequired);
  }
  if (ImGui::Button("Capture & Match"))
    m_captureRequested = true;
  ImGui::SameLine();
  ImGui::Checkbox("Dump PNGs", &m_debugDumpPNGs);
  ImGui::Text("Snapshots: %u completed, %u dropped, %zu in flight",
    m_debugCompletedSnapshots.load(), m_debugDroppedSnapshots, m_dumpRing ? m_dumpRing->outstanding() : 0);

  std::lock_guard<std::mutex> guard(m_matchStatsLock);
  for (size_t viewIdx = 0; viewIdx < m_matchStats.size(); ++viewIdx) {
    const ViewMatchStats& stats = m_matchStats[viewIdx];
    if (stats.detectedCount == 0 && !stats.valid)
      continue;
    ImGui::PushID(static_cast<int>(viewIdx));
    ImGui::Text("View %zu [seq %04u]: %u det / %u seed / %u match / %u inlier",
      viewIdx, stats.snapshot.sequence, stats.detectedCount, stats.seededCount, stats.matchedCount, stats.inlierCount);
    ImGui::Text("  dY mean %+.3f px, RMS %.3f px, coverage %u/%u, %.1f ms",
      stats.meanDeltaY, stats.rmsDeltaY, stats.coverageCells, kCoverageGridSize * kCoverageGridSize, stats.matchMilliseconds);

    const SolveResult& solve = stats.solve;
    if (solve.attempted) {
      ImGui::Text("  solve G [%+.4f %+.4f %+.4f] deg, RMS %.3f -> %.3f px",
        glm::degrees(solve.rectRotation.x), glm::degrees(solve.rectRotation.y), glm::degrees(solve.rectRotation.z),
        solve.rmsBeforePixels, solve.rmsAfterPixels);
      ImGui::Text("  inliers %.0f%%: %s", solve.solveInlierFraction * 100.0f,
        solve.gatesPassed ? "PASS" : solve.gateFailure);
      if (solve.gatesPassed) {
        if (ImGui::Button("Apply")) {
          // Guard against the base calibration having changed since capture --
          // the candidate composes on top of the capture-time rectification.
          if (stats.snapshot.calibrationRevision == m_cameraSystem->calibrationDataRevision())
            applyCorrection(viewIdx, solve.correctionCandidate);
          else
            printf("StereoSelfCalibration: stale solve for view %zu not applied (calibration revision changed)\n", viewIdx);
        }
      }
      if (viewIdx < m_voteState.size())
        ImGui::Text("  votes: %zu/%u", m_voteState[viewIdx].votes.size(), kVotesRequired);
    } else if (solve.gateFailure[0]) {
      ImGui::Text("  solve skipped: %s", solve.gateFailure);
    }
    ImGui::PopID();
  }
}
