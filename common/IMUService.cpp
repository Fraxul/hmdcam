#include "IMUService.h"
#include "common/Timing.h"
#include "imgui/imgui.h"
#include "implot/implot.h"
#include <algorithm>
#include <limits>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

const char* kCalibrationFilename = "imuCalibration.yml";

// IMUFrame ring size must be power-of-two for the ring buffer address logic to work,
// so we enforce that by declaring log2(size).
constexpr size_t kIMUFrameRingSizeLog2 = 4; // 1<<4, or 16 frames
constexpr size_t kIMUFrameRingSize = (1 << kIMUFrameRingSizeLog2);
constexpr size_t kIMUFrameRingSizeMask = ((1 << kIMUFrameRingSizeLog2) - 1);

IMUService::IMUService() {
  m_imuFrameRing = new IMUFrame[kIMUFrameRingSize];

  loadCalibrationData();

  if (m_imuHIDEndpoint.empty()) {
    printf("IMUService(): IMU HID endpoint path is empty, can't start service thread.\n");
  } else {
    m_imuReaderThread = std::thread(&IMUService::imuReaderThreadFn, this);
  }
  // TODO: Magnetometer service thread.
}

IMUService::~IMUService() {
  m_serviceThreadShutdown.store(true);

  if (m_imuReaderThread.joinable()) {
    m_imuReaderThread.join();
  }

  delete[] m_imuFrameRing;
}

#define readNode(node, settingName) cv::read(node[#settingName], m_##settingName, m_##settingName)
bool IMUService::loadCalibrationData() {
  cv::FileStorage fs(kCalibrationFilename, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    printf("Unable to open calibration data file\n");
    return false;
  }

  try {
    readNode(fs, accelMicroGPerLSB);
    readNode(fs, gyroMicroDPSPerLSB);
    readNode(fs, imuHIDEndpoint);

  } catch (const std::exception& ex) {
    printf("Unable to load calibration data: %s\n", ex.what());
    return false;
  }
  return true;
}
#undef readNode

#define writeNode(fileStorage, settingName) fileStorage.write(#settingName, m_##settingName)
void IMUService::saveCalibrationData() {
  cv::FileStorage fs(kCalibrationFilename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);

  writeNode(fs, accelMicroGPerLSB);
  writeNode(fs, gyroMicroDPSPerLSB);
  writeNode(fs, imuHIDEndpoint);
}
#undef writeNode

void IMUService::processFrame(uint64_t lastCaptureTimestampNs) {
  // Find the IMU frame that best matches the provided capture timestamp.
  // Since we only call processFrame() once per frame with increasing timestamps,
  // this should be a frame that we haven't delivered before -- one with isApproximateTimestamp set.

  int64_t minTSDelta = std::numeric_limits<int64_t>::max();
  int32_t selectedRingIdx = -1;
  for (uint32_t i = 0; i < kIMUFrameRingSize; ++i) {
    IMUFrame& frame = m_imuFrameRing[i];
    if (!frame.isApproximateTimestamp) {
      continue; // We've handed this one out before.
    }
    if (frame.validSampleCount == 0 || frame.frameStartTimestampNs == 0) {
      continue; // No valid data.
    }
    int64_t delta = std::abs(static_cast<int64_t>(frame.frameStartTimestampNs) - static_cast<int64_t>(lastCaptureTimestampNs));

    if (delta < minTSDelta) {
      // Better candidate than the previous one.
      minTSDelta = delta;
      selectedRingIdx = i;
    }
  }

  if (selectedRingIdx == -1) {
    // No valid candidate.
    m_currentIMUFrame = nullptr;
    return;
  }

  // Select current frame and lock in the timestamp
  m_currentIMUFrame = &(m_imuFrameRing[selectedRingIdx]);
  m_currentIMUFrame->frameStartTimestampNs = lastCaptureTimestampNs;
  m_currentIMUFrame->isApproximateTimestamp = false;
}

void IMUService::imuReaderThreadFn() {

  // Raw sample payload received over USB.
  struct IMUHIDSample {
    // Timestamp as a line offset (with fractional component) since start-of-frame.
    // Frame boundary is implicit from lineOffset wrapping around and being less than the previous sample.
    uint32_t lineOffset;
    // Raw gyroscope and accelerometer values. They need to be scaled by the scale factors in our config.
    int16_t gyro[3];
    int16_t accel[3];
  };
  static_assert(sizeof(IMUHIDSample) == 16);
  struct IMUHIDPacket {
    IMUHIDSample sample[4];
  };
  static_assert(sizeof(IMUHIDPacket) == 64);


  int fd = -1;
  uint32_t lastLineOffset = 0;
  int lastOpenErrno = 0;

  // Frame that we're in the process of building
  IMUFrame* currentFrame = &(m_imuFrameRing[m_imuFrameRingWriteIdx & kIMUFrameRingSizeMask]);

  while (!m_serviceThreadShutdown) {
    if (fd < 0) {
      fd = ::open(m_imuHIDEndpoint.c_str(), O_RDONLY | O_CLOEXEC);

      if (fd < 0) {
        // Open failed -- retry in a sec.
        if (lastOpenErrno != errno) {
          // Report error if it's not the last one we saw (avoids log spam)
          lastOpenErrno = errno;
          printf("IMUService::imuReaderThreadFn(): open(%s): %s (%d)\n", m_imuHIDEndpoint.c_str(), strerror(lastOpenErrno), lastOpenErrno);
        }
        delayMs(1000);
        continue;
      } else {
        // Opened OK
        printf("IMUService::imuReaderThreadFn(): Opened HID endpoint %s\n", m_imuHIDEndpoint.c_str());
        lastOpenErrno = 0;
      }
    }

    IMUHIDPacket packet;
    int res = read(fd, &packet, sizeof(packet));
    // Grab a packet timestamp as close to the read() return as possible.
    // Won't be completely accurate due to scheduling jitter, but it should be close enough
    // to align the camera timestamps with the IMU frame boundary.
    uint64_t packetTimestamp = currentTimeNs();
    if (res <= 0) {
      // Close and attempt to re-open on read error
      printf("IMUService::imuReaderThreadFn(): read error: %s (%d)\n", strerror(errno), errno);
      ::close(fd);
      fd = -1;
      continue;
    }
    if (res != sizeof(packet)) {
      printf("IMUService::imuReaderThreadFn(): bad read length: expected %zud, got %d\n", sizeof(packet), res);
      continue;
    }

    for (size_t sampleIdx = 0; sampleIdx < 4; ++sampleIdx) {
      const IMUHIDSample& hidSample = packet.sample[sampleIdx];

      // For each packet:
      // If the timestamp rolled over, then we commit the active frame and start a new one.
      if (hidSample.lineOffset < lastLineOffset) {
        // Commit active frame and start a new frame.
        m_imuFrameRingWriteIdx += 1;
        currentFrame = &(m_imuFrameRing[m_imuFrameRingWriteIdx & kIMUFrameRingSizeMask]);
        currentFrame->resetWithApproximateTimestamp(packetTimestamp);
      }
      lastLineOffset = hidSample.lineOffset;

      // Push the sample onto the current frame.
      // (Simple append, since we're guaranteed that timestamps will never go backwards).
      if (currentFrame->validSampleCount < kMaxIMUSamplesPerFrame) {
        IMUSample& outSample = currentFrame->samples[currentFrame->validSampleCount++];

        // lineOffset doesn't require a unit conversion, just copy it over.
        outSample.lineOffset = hidSample.lineOffset;

        // Apply accelerometer and gyroscope scaling
        for (size_t i = 0; i < 3; ++i) {
          outSample.gyroDPS[i] = static_cast<float>(hidSample.gyro[i] * m_gyroMicroDPSPerLSB) / 1000000.0f;
          outSample.accelG[i] = static_cast<float>(hidSample.accel[i] * m_accelMicroGPerLSB) / 1000000.0f;
        }
      }
    }
  }
  // Shutdown requested. Cleanup IMU FD first.
  if (fd >= 0) {
    ::close(fd);
  }
}

void IMUService::renderIMGUI() {
  // TODO: graphs
  if (m_currentIMUFrame && m_currentIMUFrame->validSampleCount) {
    const IMUSample& sample = m_currentIMUFrame->samples[0];
    ImGui::Text("Accel: %.3f %.3f %.3f\nGyro: %.3f %.3f %.3f",
      sample.accelG[0],
      sample.accelG[1],
      sample.accelG[2],
      sample.gyroDPS[0],
      sample.gyroDPS[1],
      sample.gyroDPS[2]);
  } else {
    ImGui::Text("No IMU frame!");
  }
}
