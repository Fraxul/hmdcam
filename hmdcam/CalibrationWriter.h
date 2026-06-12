#pragma once
#include "common/IMUService.h"
#include "IArgusCamera.h"
#include <cstdint>
#include <queue>
#include <mutex>
#include <vector>

class CalibrationWriter {
public:
  CalibrationWriter(IArgusCamera*);
  ~CalibrationWriter();

  void processFrame(IMUFrame*);

  void renderIMGUI();

  struct FrameData {
    FrameData() = default;
    ~FrameData() {
      cuMemFreeHost(cudaHostPtr);
      cuEventDestroy(copyDoneEvent);
    }

    FrameData(const FrameData&) = delete;
    FrameData& operator=(const FrameData&) = delete;

    void* cudaHostPtr = nullptr;
    CUevent copyDoneEvent = 0;
    uint32_t width, height;
    uint64_t timestamp;
    int dirFd;
  };

protected:
  struct StreamData {
    bool enabled = true; // Whether this stream is enabled for capture. Don't change this flag while capture is running.
    int dirFd = -1; // dirFd for the output directory for this stream.
  };

  void asyncWriteFrameData(FrameData*);

  void setActive(bool);

  // Closes all output directory/file descriptors and resets them to -1.
  // Idempotent: safe to call on a partially-initialized state.
  void closeOutputDescriptors();

  // Configuration. Don't change this while active.
  bool m_triggered = false; // For trigger mode
  uint32_t m_writeInterval = 1; // Write every n frames; 0 is trigger-only mode.
  bool m_writeImuOnly = false; // Skip streams, only write IMU
  std::vector<StreamData> m_streamData;

  // State.
  bool m_active = false;
  bool m_inShutdown = false; // Transition state when we're deactivating -- drains work from FrameData queue, then calls closeOutputDescriptors(). Will only be true when m_active is true.
  uint32_t m_frameIndexCounter = 0;

  int m_dirFd = -1; // Output directory fd
  int m_imuFd = -1; // IMU output fd

  uint32_t m_writtenFrames = 0;
  uint32_t m_droppedFrames = 0;
  uint64_t m_baseTimestampOffset = 0;

  FrameData* m_frameData = nullptr;
  uint32_t m_frameDataCount = 0;

  std::queue<FrameData*> m_frameDataFreeList;
  std::mutex m_frameDataFreeListLock;

  IArgusCamera* cameraProvider() const { return m_cameraProvider; }
  IArgusCamera* m_cameraProvider;
};
