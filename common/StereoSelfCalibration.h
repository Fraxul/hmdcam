#pragma once
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>
#include <glm/gtc/quaternion.hpp>
#include <opencv2/core.hpp>

class AsyncGpuDumpRing;
class CameraSystem;
class DepthMapGenerator;
struct IMUFrame;

// Online stereo self-calibration.
//
// Periodically measures residual vertical disparity between a view's rectified
// stereo pair and solves for a small correction to the pair's relative camera
// rotation, applied via CameraSystem::View::stereoCorrectionTarget.
//
// Works vis snapshot capture + sparse feature matching. The rectified luma pair
// and the post-processed disparity map are async-copied into pinned host memory.
// A worker thread detects corners in the left image and matches them into the
// right image with pyramidal LK, seeded horizontally from the SGM disparity
// map. The per-match vertical residuals (deltaY) are the measurement the
// correction solve will consume in the next stage.
class StereoSelfCalibration {
public:
  StereoSelfCalibration(CameraSystem*, DepthMapGenerator*);
  ~StereoSelfCalibration();

  // Processing should be called immediately following DepthMapGenerator::processFrame().
  // The (optional) IMU frame drives the automatic-capture stability gate.
  // Also collects completed async solve results and updates CameraSystem calibration data.
  void processFrame(IMUFrame*);

  // ImGui debug controls
  void renderIMGUI();

protected:
  // Everything needed to interpret a slot's pinned buffer once the async
  // copies complete, plus the capture-time state the solver stage will need.
  struct SnapshotMetadata {
    size_t viewIdx = 0;
    uint32_t sequence = 0;

    // Tightly-packed plane layout within the slot buffer.
    size_t lumaOffset[2] = {0, 0}; // [left, right]
    uint32_t lumaWidth = 0, lumaHeight = 0;
    size_t disparityOffset = 0;
    uint32_t disparityWidth = 0, disparityHeight = 0;
    int disparityCvType = 0;
    float disparityPrescale = 1.0f; // raw disparity -> pixel units
    uint32_t maxDisparityRaw = 0;

    // Capture-time system state. The solver's output is a residual on top of
    // correctionAtCapture; a mismatched calibrationRevision invalidates the
    // snapshot entirely.
    glm::quat correctionAtCapture = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    unsigned int calibrationRevision = 0;

    // Right-eye rectification geometry at capture, row-major 3x3, for the
    // correction solve. The projection is at camera stream resolution;
    // streamWidth/Height convert luma-resolution match coordinates into its
    // pixel space.
    double stereoProjectionRight[9] = {0};
    double stereoRectificationRight[9] = {0};
    uint32_t streamWidth = 0, streamHeight = 0;
  };

  // Output of the correction solve over one snapshot's matches.
  struct SolveResult {
    bool attempted = false; // match stats were good enough to try
    bool converged = false;
    bool gatesPassed = false;
    char gateFailure[48] = {}; // first failed gate, for the debug UI

    // Solved rectified-frame rotation G (angle-axis, radians) -- the raw
    // optimizer output, useful for display: [pitch, yaw, roll] for small angles.
    glm::vec3 rectRotation = glm::vec3(0.0f);

    // Full replacement value for View::stereoCorrectionTarget
    // (= correctionAtCapture composed with the solved residual).
    glm::quat correctionCandidate = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

    float rmsBeforePixels = 0.0f; // vertical residual over inliers, stream-res pixels
    float rmsAfterPixels = 0.0f;
    float solveInlierFraction = 0.0f; // |residual| < kSolveInlierThresholdPixels after solve
    float solveMilliseconds = 0.0f;
  };

  // Feature coverage is scored on a kCoverageGridSize^2 occupancy grid over
  // the luma image; the solve gate will require spread in both axes.
  static constexpr uint32_t kCoverageGridSize = 4;

  // Result of the sparse feature-match pass over one snapshot. leftPoints /
  // rightPoints hold the surviving inlier pairs in rectified luma pixels; the
  // vertical residual of match i is rightPoints[i].y - leftPoints[i].y.
  struct ViewMatchStats {
    SnapshotMetadata snapshot;
    bool valid = false; // at least one inlier survived

    uint32_t detectedCount = 0; // corners from goodFeaturesToTrack
    uint32_t seededCount = 0; // corners with a usable disparity seed
    uint32_t matchedCount = 0; // survived forward + backward LK
    uint32_t inlierCount = 0; // survived the |deltaY| prefilter
    float meanDeltaY = 0.0f; // pixels, over inliers
    float rmsDeltaY = 0.0f; // pixels, over inliers
    uint32_t coverageMask = 0; // kCoverageGridSize^2 occupancy bits, bit = (cellY * grid) + cellX
    uint32_t coverageCells = 0; // popcount of coverageMask
    float matchMilliseconds = 0.0f;

    std::vector<cv::Point2f> leftPoints, rightPoints;

    SolveResult solve;
  };

  bool captureView(size_t viewIdx); // false if the snapshot was dropped

  // Ring worker thread: match features, publish stats, optionally dump PNGs.
  void processSnapshot(const SnapshotMetadata&, const uint8_t* slotBase, bool dumpPNGs);
  void matchFeatures(const SnapshotMetadata&, const cv::Mat& leftLuma, const cv::Mat& rightLuma, const cv::Mat& disparityPixels, ViewMatchStats& outResult) const;
  void solveCorrection(ViewMatchStats&) const; // fills outResult.solve from the match inliers
  void debugDumpSnapshot(const SnapshotMetadata&, const uint8_t* slotBase);
  void debugDumpMatchAnnotations(const cv::Mat& leftLuma, const ViewMatchStats&);

  // ----- Automatic calibration (render thread only) -----

  // Consume newly-completed solve results into the per-view vote buffers and
  // auto-apply once enough consecutive candidates agree.
  void consumeSolveResults();

  // Returns true when the IMU reports the rig quiescent (or when no IMU data
  // is available, in which case the gate is disabled rather than blocking).
  bool isIMUStable(IMUFrame*) const;

  // Sets the view's stereoCorrectionTarget and schedules a sidecar save.
  void applyCorrection(size_t viewIdx, const glm::quat&);

  void loadSavedCorrections(); // ctor: prime active+target from onlineCalibration.yml
  void saveCorrections() const; // async rewrite of onlineCalibration.yml

  // Consecutive gates-passed candidates for one view. A candidate that
  // disagrees with the buffered votes restarts the buffer; scene-quality gate
  // failures leave it alone (a blank wall shouldn't discard good votes).
  struct VoteState {
    int64_t lastConsumedSequence = -1;
    std::vector<glm::quat> votes;
  };
  std::vector<VoteState> m_voteState;

  bool m_autoCalibrationEnabled = true;
  float m_autoIntervalSeconds = 5.0f;
  float m_gyroStabilityThresholdDPS = 3.0f;
  uint32_t m_consecutiveStableFrames = 0;
  std::chrono::steady_clock::time_point m_lastAutoCaptureTime;

  CameraSystem* m_cameraSystem;
  DepthMapGenerator* m_depthMapGenerator;

  AsyncGpuDumpRing* m_dumpRing = nullptr; // Created lazily once buffer sizes are known.
  uint32_t m_captureSequence = 0;

  // Latest match result per view. Written by ring worker threads, read by the
  // UI (and later, the solve stage).
  mutable std::mutex m_matchStatsLock;
  std::vector<ViewMatchStats> m_matchStats;

  // Debug UI state
  bool m_captureRequested = false;
  bool m_debugDumpPNGs = false;
  uint32_t m_debugDroppedSnapshots = 0; // render thread only
  std::atomic<uint32_t> m_debugCompletedSnapshots{0}; // incremented on ring worker threads

private:
  StereoSelfCalibration(const StereoSelfCalibration&);
  StereoSelfCalibration& operator=(const StereoSelfCalibration&);
};
