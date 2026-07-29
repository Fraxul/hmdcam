#pragma once
#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include <glm/glm.hpp>

constexpr size_t kMaxIMUSamplesPerFrame = 64;


// Contains one sample in an IMU frame.
// Sample timing is expressed entirely in the sync controller's line-offset timebase -- the
// same XVS/XHS-derived counter that clocks the camera readout -- so samples carry no
// wall-clock timestamps. To integrate rate data, convert a line-offset delta to seconds
// with IMUFrame::tickDurationNs.
struct IMUSample {
  uint32_t lineOffset; // Line offset (with 8-bit fractional component) since start-of-frame.
  glm::vec3 gyroDPS; // degrees/sec
  glm::vec3 accelG; // G
};

// Contains one set of IMU samples for a frame. validSampleCount is variable.
struct IMUFrame {
  // Start-of-frame timestamp, currentTimeNs() timebase. In the internal frame ring this is
  // approximate (derived from HID packet arrival time) and is only used to associate the
  // frame with a capture; the snapshot handed out by IMUService::currentIMUFrame() has it
  // snapped to the matched capture's exact timestamp.
  uint64_t frameStartTimestampNs = 0;

  // Monotonic count of frames committed by the reader thread. 0 = slot never written.
  uint64_t commitSequence = 0;

  uint32_t validSampleCount = 0;

  // Duration of one lineOffset tick (1/256th of a sensor readout line), from configuration.
  double tickDurationNs = 0.0;

  IMUSample samples[kMaxIMUSamplesPerFrame];
};

class IMUService {
public:
  IMUService();
  ~IMUService();

  // Call processFrame() once per capture to associate a committed IMU frame with the
  // capture timestamp and update currentIMUFrame().
  void processFrame(uint64_t lastCaptureTimestampNs);

  // Returns a pointer to a consumer-owned snapshot of the IMU frame matched to the capture
  // timestamp most recently provided to processFrame(). The snapshot is stable until the
  // next processFrame() call -- the reader thread never touches it. Can be NULL if there
  // was no timestamp match or if the IMU is unavailable.
  IMUFrame* currentIMUFrame() const { return m_currentIMUFrame; }

  // Load configuration data.
  bool loadConfiguration();
  void saveConfiguration();

  void renderIMGUI();

protected:
  int openSyncControllerEndpoint(uint8_t endpointIdx);

  void imuReaderThreadFn();
  std::thread m_imuReaderThread;
  // std::thread m_magHidServiceThread;
  std::atomic_bool m_serviceThreadShutdown = false;

  // IMU frame ring. Slots are written only by the reader thread (under m_frameRingLock)
  // when a fully-built frame is committed; processFrame() copies the matched slot out
  // under the same lock, so consumers never hold references into the ring.
  IMUFrame* m_imuFrameRing = nullptr;
  uint32_t m_imuFrameRingWriteIdx = 0; // reader thread only
  uint64_t m_frameCommitCounter = 0; // reader thread only; published via IMUFrame::commitSequence
  std::mutex m_frameRingLock;

  // Consumer-side state, updated in processFrame()
  IMUFrame m_currentFrameSnapshot;
  IMUFrame* m_currentIMUFrame = nullptr; // &m_currentFrameSnapshot when a frame matched, else nullptr
  uint64_t m_lastDeliveredCommitSequence = 0;

  // Capture-interval rolling average. Used for diagnostic display and to size the
  // frame-match gate in processFrame() -- sample timing is derived from line offsets and
  // never touches this.
  static constexpr uint32_t kFrameIntervalSampleCount = 64;
  uint64_t m_averageFrameIntervalNs = 0;
  uint64_t m_previousCaptureTimestampNs = 0;

  // Configuration data
  int32_t m_accelMicroGPerLSB = 244; // Default: LSM6DS3 CTRL1_XL_SCALE = 0b11 / 8g full-scale
  int32_t m_gyroMicroDPSPerLSB = 35000; // Default: LSM6DS3 CTRL2_G_SCALE = 0b100 / 1000 DPS full-scale
  int32_t m_imuTimestampTicksPerFrame = 1250 * 256; // Default: IMX662 1250-line readout with 8-bit sub-line precision, as reported by our sync controller.
  double m_imuTimestampTickDurationNs = 11111111.0 / (1250.0 * 256.0); // Default: IMX662 1250-line frame at 90fps.

  std::string m_imuHIDEndpoint; // Path to hidraw endpoint for IMU data streaming

  // Count of HID samples dropped by the line-offset sanity gate. Reader thread increments.
  std::atomic<uint32_t> m_rejectedSampleCount = 0;
};
