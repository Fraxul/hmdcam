#include "CalibrationWriter.h"
#include "common/PosixFileUtil.h"
#include "imgui/imgui.h"
#include "rhi/cuda/RHICUDA.h"
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

CalibrationWriter::CalibrationWriter(IArgusCamera* _cameraProvider) :
  m_streamWidth(_cameraProvider->streamWidth()),
  m_streamHeight(_cameraProvider->streamHeight()),
  // Each slot holds a tightly-packed 8-bit greyscale plane. The device->host copy in
  // processFrame() repacks the (pitch-padded) source into this contiguous layout, so we can write
  // it straight to disk as a raw PGM with no per-row handling. Size the pool for a few frames'
  // worth of in-flight work across all streams.
  m_dumpRing(_cameraProvider->streamCount() * 4, static_cast<size_t>(_cameraProvider->streamWidth()) * _cameraProvider->streamHeight()),
  m_cameraProvider(_cameraProvider) {

  m_streamData.resize(m_cameraProvider->streamCount());
}

CalibrationWriter::~CalibrationWriter() {
  if (m_active) {
    // Shutdown and drain immediately. We must block until all in-flight writes finish before
    // closing the output fds they reference (and before the ring frees its buffers).
    setActive(false);
    m_dumpRing.drainBlocking();

    closeOutputDescriptors();
    m_active = false;
    m_inShutdown = false;
  }
}

void CalibrationWriter::processFrame(IMUFrame* imuFrame) {

  if (!m_active)
    return;

  if (m_inShutdown) {
    // We're trying to shut down the output right now.
    // Wait until all in-flight writes have finished, then complete the shutdown.
    if (m_dumpRing.outstanding() != 0)
      return; // Work is still pending.

    printf("CalibrationWriter::processFrame(): Work queue drained -- shutdown complete.\n");
    closeOutputDescriptors();
    m_active = false;
    m_inShutdown = false;
    return;
  }

  // Compute timestamp for this capture
  uint64_t ts = cameraProvider()->frameTimestamp();
  if (m_baseTimestampOffset == 0) {
    // First-captured frame -- timestamp base starts at zero.
    m_baseTimestampOffset = ts;
    ts = 0;
  } else {
    // Offset subsequent frames by the timestamp of the first-captured frame.
    ts -= m_baseTimestampOffset;
  }

  // We always write IMU samples every frame, even in frame-decimation mode.
  // Continuous samples will be better for any downstream calibration process that consumes them,
  // and they're pretty cheap to write.
  if (m_imuFd >= 0 && imuFrame) {
    for (uint32_t sampleIdx = 0; sampleIdx < imuFrame->validSampleCount; ++sampleIdx) {
      const IMUSample& sample = imuFrame->samples[sampleIdx];

      char sampleBuf[256];
      // timestamp,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,line_offset
      // timestamp is in nanoseconds, derived here from the frame capture timestamp plus
      //   the sample's line offset (samples themselves carry only line offsets)
      // gyro_xyz are in deg/sec
      // accel_xyz are in G
      // line_offset is in sync-controller ticks (sensor readout lines with an 8-bit fraction)

      const uint64_t sampleTimestampNs = imuFrame->frameStartTimestampNs +
        static_cast<uint64_t>(static_cast<double>(sample.lineOffset) * imuFrame->tickDurationNs);
      int64_t sampleTs = static_cast<int64_t>(sampleTimestampNs) - static_cast<int64_t>(m_baseTimestampOffset);
      if (sampleTs < 0) {
        continue; // Sample timestamp should never go negative
      }

      // Timestamp here is zero-padded to 16 digits, same as the filenames.
      int sampleLen = snprintf(sampleBuf, sizeof(sampleBuf), "%016ld,%f,%f,%f,%f,%f,%f,%u\n",
        sampleTs,
        sample.gyroDPS[0], sample.gyroDPS[1], sample.gyroDPS[2],
        sample.accelG[0], sample.accelG[1], sample.accelG[2],
        sample.lineOffset);

      writeFully(m_imuFd, sampleBuf, sampleLen);
    }
  }

  if (m_writeInterval == 0) {
    // Triggered mode
    if (!m_triggered)
      return;

    // Trigger was active; reset it, then continue with capture.
    m_triggered = false;
  } else {
    // Every-N-frames mode
    ++m_frameIndexCounter;
    if (m_frameIndexCounter >= m_writeInterval) {
      // Roll over counter and capture frame
      m_frameIndexCounter = 0;
    } else {
      return;
    }
  }

  bool didDropFrame = false;
  // Capture and write frame.
  if (!m_writeImuOnly) {

    for (size_t streamIdx = 0; streamIdx < cameraProvider()->streamCount(); ++streamIdx) {
      if (!m_streamData[streamIdx].enabled)
        continue; // Capture not enabled on this stream.

      // Grab a slot from the dump ring.
      AsyncGpuDumpRing::Slot* slot = m_dumpRing.acquire();
      if (!slot) {
        // We're not draining the ring fast enough -- can't capture this frame.
        didDropFrame = true;
        break;
      }

      // Issue CUDA copy
      CUDA_MEMCPY2D copyDescriptor;
      memset(&copyDescriptor, 0, sizeof(copyDescriptor));
      cameraProvider()->fillCudaMemcpy2DForStreamSource(copyDescriptor, streamIdx, /*fromChromaPlane=*/ false);

      copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
      copyDescriptor.dstHost = slot->hostPtr;
      copyDescriptor.dstPitch = m_streamWidth; // tightly-packed host destination (no pitch padding)
      CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, RHICUDA::defaultAsyncStream));

      // Record copy completion event
      CUDA_CHECK(cuEventRecord(slot->copyDoneEvent, RHICUDA::defaultAsyncStream));

      // Queue async work to write the frame to disk. The destination dir and the filename timestamp
      // are per-capture policy, so we bind them into the write callback.
      int dirFd = m_streamData[streamIdx].dirFd; // per-camera output dirFd
      m_dumpRing.dispatch(slot, [this, dirFd, ts](AsyncGpuDumpRing::Slot* writeSlot) {
        writePGMFrame(writeSlot, dirFd, ts);
      });
    }
  }

  if (didDropFrame)
    m_droppedFrames += 1;
  else
    m_writtenFrames += 1;
}

void CalibrationWriter::writePGMFrame(AsyncGpuDumpRing::Slot* slot, int dirFd, uint64_t timestamp) {
  // The dump ring has already waited on the copy-completion event before invoking us, and will
  // recycle the slot once we return.

  // Generate filename with nanosecond timestamp, zero-padded so it sorts nicely
  char filename[128];
  snprintf(filename, 128, "%016lu.pgm", timestamp);

  // Write to disk using filename and dirFd.
  // We emit a binary PGM (P5): a short ASCII header followed by the tightly-packed 8-bit greyscale plane.
  // PNG compression is too CPU-intensive on this platform to keep up with the "all cameras, every frame" data rate.
  int fd = openat(dirFd, filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd >= 0) {
    char header[64];
    int headerLength = snprintf(header, sizeof(header), "P5\n%u %u\n255\n", m_streamWidth, m_streamHeight);
    size_t pixelBytes = static_cast<size_t>(m_streamWidth) * m_streamHeight;

    if (!writeFully(fd, header, headerLength) || !writeFully(fd, slot->hostPtr, pixelBytes))
      printf("CalibrationWriter::writePGMFrame(): write() error: %s\n", strerror(errno));

    close(fd);
  } else {
    printf("CalibrationWriter::writePGMFrame(): openat() error: %s\n", strerror(errno));
  }
}

void CalibrationWriter::setActive(bool active) {
  if (m_active == active)
    return;

  // Refuse to change states during shutdown.
  if (m_inShutdown)
    return;

  if (active) {
    // Need to create a calibration directory and open the dir fd
    char dirName[256];

    std::time_t now = std::time(nullptr);
    std::strftime(dirName, sizeof(dirName), "calibration-data-%Y%m%d-%H%M%S", std::localtime(&now));

    if (mkdir(dirName, 0777) != 0) {
      printf("CalibrationWriter::setActive(): Can't create calibration directory \"%s\": %s\n", dirName, strerror(errno));
      return;
    }

    m_dirFd = open(dirName, O_DIRECTORY | O_PATH);
    if (m_dirFd < 0) {
      printf("CalibrationWriter::setActive(): Can't open calibration directory \"%s\": %s\n", dirName, strerror(errno));
      return;
    }

    // Create per-stream output directories
    for (size_t streamIdx = 0; streamIdx < m_streamData.size(); ++streamIdx) {
      if (!m_streamData[streamIdx].enabled || m_writeImuOnly)
        continue; // Don't bother making directories for disabled streams

      snprintf(dirName, 256, "camera%zu", streamIdx);
      if (mkdirat(m_dirFd, dirName, 0777) != 0) {
        printf("CalibrationWriter::setActive(): Can't create calibration directory \"%s\": %s\n", dirName, strerror(errno));
        closeOutputDescriptors();
        return;
      }

      m_streamData[streamIdx].dirFd = openat(m_dirFd, dirName, O_DIRECTORY | O_PATH);
      if (m_streamData[streamIdx].dirFd < 0) {
        printf("CalibrationWriter::setActive(): Can't open calibration directory \"%s\": %s\n", dirName, strerror(errno));
        closeOutputDescriptors();
        return;
      }
    }

    // Open IMU output log.
    // We only write IMU samples in continous-output mode; they're probably not useful in triggered mode.
    if (m_writeInterval != 0 || m_writeImuOnly) {
      m_imuFd = openat(m_dirFd, "imu.csv", O_WRONLY | O_CREAT | O_TRUNC, 0644);
      if (m_imuFd < 0) {
        printf("CalibrationWriter::setActive(): Can't open IMU log file: %s\n", strerror(errno));
        closeOutputDescriptors();
        return;
      }

      // Write CSV format header.
      // timestamp is in nanoseconds (derived from the capture timestamp + line offset)
      // gyro_xyz are in deg/sec
      // accel_xyz are in G
      // line_offset is in sync-controller ticks (readout lines with an 8-bit fraction)
      const char* imuCSVHeader = "timestamp,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,line_offset\n";
      write(m_imuFd, imuCSVHeader, strlen(imuCSVHeader));
    }

    // Reset stats counters
    m_writtenFrames = 0;
    m_droppedFrames = 0;
    m_frameIndexCounter = 0;

    // Timestamp restarts at zero for each capture.
    m_baseTimestampOffset = 0;

    // Finally, set the active flag so we start writing.
    m_active = true;
    m_inShutdown = false;
  } else {
    // Set the shutdown flag so we stop writing.
    // processFrame() will finish the shutdown and clear m_active once all async write tasks have finished.
    printf("CalibrationWriter::setActive(): Wrote %u frames, dropped %u frames\n", m_writtenFrames, m_droppedFrames);
    m_inShutdown = true;
  }
}

void CalibrationWriter::closeOutputDescriptors() {
  if (m_dirFd >= 0) {
    close(m_dirFd);
    m_dirFd = -1;
  }

  if (m_imuFd >= 0) {
    close(m_imuFd);
    m_imuFd = -1;
  }

  for (size_t streamIdx = 0; streamIdx < m_streamData.size(); ++streamIdx) {
    int& fd = m_streamData[streamIdx].dirFd;
    if (fd >= 0) {
      close(fd);
      fd = -1;
    }
  }
}

void CalibrationWriter::renderIMGUI() {
  ImGui::PushID(this);

  if (m_active) {
    ImGui::BeginDisabled(m_inShutdown);
    if (m_writeInterval == 0) {
      if (ImGui::Button("Trigger Capture")) {
        m_triggered = true;
      }
    }
    if (ImGui::Button("Stop Writing Calibration Data")) {
      setActive(false);
    }
    ImGui::EndDisabled();
    ImGui::Text("Written: %u || Dropped: %u", m_writtenFrames, m_droppedFrames);
  } else {
    if (ImGui::Button("Start Writing Calibration Data")) {
      setActive(true);
    }
  }

  ImGui::BeginDisabled(m_active);
  ImGui::Checkbox("IMU data only", &m_writeImuOnly);

  ImGui::BeginDisabled(m_writeImuOnly);
  for (size_t streamIdx = 0; streamIdx < m_streamData.size(); ++streamIdx) {
    char buf[64];
    snprintf(buf, 64, "Enable stream %zu", streamIdx);
    ImGui::Checkbox(buf, &m_streamData[streamIdx].enabled);
  }

  ImGui::SeparatorText("Trigger/Frequency");
  if (ImGui::RadioButton("Save on Trigger", m_writeInterval == 0)) m_writeInterval = 0;
  if (ImGui::RadioButton("Every frame", m_writeInterval == 1)) m_writeInterval = 1;
  if (ImGui::RadioButton("Every 2 frames", m_writeInterval == 2)) m_writeInterval = 2;
  if (ImGui::RadioButton("Every 3 frames", m_writeInterval == 3)) m_writeInterval = 3;
  ImGui::EndDisabled(); // m_writeImuOnly
  ImGui::EndDisabled(); // m_active

  ImGui::PopID();
}
