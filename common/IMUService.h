#pragma once
#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>

constexpr size_t kMaxIMUSamplesPerFrame = 64;


// Contains one sample in an IMU frame
struct IMUSample {
  uint32_t lineOffset; // Tiemstamp as a line offset (with fractional component) since start-of-frame.
  float gyroDPS[3]; // XYZ, degrees/sec
  float accelG[3]; // XYZ, G
};

// Contains one set of IMU samples for a frame. validSampleCount is variable.
struct IMUFrame {
  void resetWithApproximateTimestamp(uint64_t ts) {
    frameStartTimestampNs = ts;
    validSampleCount = 0;
    isApproximateTimestamp = true;
  }

  uint64_t frameStartTimestampNs = 0; // currentTimeNs timebase
  uint32_t validSampleCount = 0;

  // isApproximateTimestamp will initially be true while the timestamp is a guess based on the USB frame arrival time.
  // When we receive a camera frame timestamp that aligns with this IMU frame, we re-write the base timestamp.
  // isApproximateTimestamp will then be false.
  bool isApproximateTimestamp = true;

  IMUSample samples[kMaxIMUSamplesPerFrame];
};

class IMUService {
public:
  IMUService();
  ~IMUService();

  // Call processFrame() to provide concrete capture timestamps for sample timing and update current* accessors.
  void processFrame(uint64_t lastCaptureTimestampNs);

  // Returns a non-owning pointer to the IMUFrame associated with the frame-timestamp most recently provided to processFrame().
  // Can be NULL if there was no timestamp match or if the IMU is unavailable.
  IMUFrame* currentIMUFrame() const { return m_currentIMUFrame; }

  bool loadCalibrationData();
  void saveCalibrationData();

  void renderIMGUI();

protected:
  int openSyncControllerEndpoint(uint8_t endpointIdx);

  void imuReaderThreadFn();
  std::thread m_imuReaderThread;
  // std::thread m_magHidServiceThread;
  std::atomic_bool m_serviceThreadShutdown = false;

  // IMU frame ring
  IMUFrame* m_imuFrameRing = nullptr;
  uint32_t m_imuFrameRingWriteIdx = 0;

  // Current-frame state, updated in processFrame()
  IMUFrame* m_currentIMUFrame = nullptr; // non-owning pointer into m_imuFrameRing

  // Calibration data
  int32_t m_accelMicroGPerLSB = 244; // Default: LSM6DS3 CTRL1_XL_SCALE = 0b11 / 8g full-scale
  int32_t m_gyroMicroDPSPerLSB = 35000; // Default: LSM6DS3 CTRL2_G_SCALE = 0b100 / 1000 DPS full-scale

  std::string m_imuHIDEndpoint; // Path to hidraw endpoint for IMU data streaming
};
