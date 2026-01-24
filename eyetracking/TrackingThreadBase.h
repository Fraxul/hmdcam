#pragma once
#include "V4L2Camera.h"
#include <stdint.h>
#include <unistd.h>
#include <boost/thread.hpp>

class TrackingThreadBase {
public:

  // Call from processFrame on the main-thread loop.
  // Does no work, but ensures that the tracking thread is running and handles (re)opening the camera.
  void processFrameHook();

  // Entry point for m_processingThread.
  void processingThreadFn();

  // Should be called during derived class dtor before destroying any state that is touched by internalProcessOneCapture.
  void shutdownThread();

  // Called once on the processing thread after opening the video capture device, before any calls to internalProcessOneCapture.
  virtual void internalUpdateStateOnCaptureOpen() = 0;
  // Called on the processing thread once per capture. Implemented in derived class.
  virtual void internalProcessOneCapture() = 0;

  virtual ~TrackingThreadBase();

  // Configuration -- things that should be set by derived classes.
  std::string m_captureDirName = "captures"; // Probably should override this.
  std::string m_captureFileSuffix = ""; // Goes between the index and the extension when constructing a filename.

  // State
  uint64_t m_lastCaptureTimestampNs = 0; // currentTimeNs

  uint64_t m_captureFileIndex = 0; // Set to non-zero to one-shot capture to a file
  bool m_debugFreezeCapture = false;

  std::string m_cameraDeviceName; // Loaded from and saved to the config.
  std::string m_cameraDeviceNameOverride; // used if empty, otherwise m_cameraDeviceName. Not saved to the config.
  const char* getCameraDeviceName() const {
    return m_cameraDeviceNameOverride.empty() ? m_cameraDeviceName.c_str() : m_cameraDeviceNameOverride.c_str();
  }

  V4L2Camera m_capture;

  bool processingThreadAlive() const { return m_processingThreadAlive; }


  // Internal state
private:
  boost::thread m_processingThread;
  bool m_processingThreadAlive = false;

  // Ratelimiting for capture-open attempts
  uint64_t m_lastCaptureOpenAttemptTimeNs = 0; // currentTimeNs
};

