#include "TrackingThreadBase.h"
#include "common/Timing.h"
#include "stb/stb_image_write.h"

TrackingThreadBase::~TrackingThreadBase() {
  // Safety net: derived dtors are expected to call shutdownThread() before destroying
  // state the thread touches, but we check again anyway to avoid std::terminate() from
  // destroying a joinable thread.
  shutdownThread();
}

void TrackingThreadBase::shutdownThread() {
  m_shutdownRequested = true;
  if (m_processingThread.joinable())
    m_processingThread.join();
}

void TrackingThreadBase::processFrameHook() {
  if (!m_processingThreadAlive) {
    if (!(m_cameraDeviceName.empty() && m_cameraDeviceNameOverride.empty())) {
      // Try re-starting processing, ratelimited to once a second
      if (deltaTimeMs(m_lastCaptureOpenAttemptTimeNs, currentTimeNs()) > 1000.0f) {
        m_lastCaptureOpenAttemptTimeNs = currentTimeNs();

        if (m_capture.tryOpenSensor(getCameraDeviceName())) {
          // Capture is open, restart the processing thread.
          // Join any prior thread that has already exited on its own (readFrame failure) before reassigning.
          // (Move-assigning over a still-joinable FxThread would call std::terminate).
          if (m_processingThread.joinable())
            m_processingThread.join();
          m_shutdownRequested = false;
          m_processingThread = FxThread(&TrackingThreadBase::processingThreadFn, this);
          printf("TrackingThreadBase: Successfully opened capture of \"%s\"\n", getCameraDeviceName());
        }
      }
    }
  }
}

void TrackingThreadBase::processingThreadFn() {
  m_processingThreadAlive = true;
  this->internalUpdateStateOnCaptureOpen();

  while (true) {
    if (m_shutdownRequested)
      break;

    // Capture frame
    if (!m_debugFreezeCapture) {
      if (!m_capture.readFrame()) {
        printf("TrackingThreadBase(%p): readFrame() returned false, terminating\n", this);
        break;
      }
    }

    m_lastCaptureTimestampNs = currentTimeNs();

    // Save capture to disk if requested
    if (m_captureFileIndex) {
      char fnbuf[256];
      snprintf(fnbuf, 255, "%s/%lu%s.png", m_captureDirName.c_str(), m_captureFileIndex, m_captureFileSuffix.c_str());
      fnbuf[255] = '\0';

      // Save without cropping
      const cv::Mat& fullCap = m_capture.lumaPlane();
      if (stbi_write_png(fnbuf, fullCap.cols, fullCap.rows, /*components=*/ fullCap.channels(), fullCap.ptr(), /*rowBytes=*/ fullCap.step)) {
        printf("TrackingThreadBase(%p): wrote capture to file %s\n", this, fnbuf);
      } else {
        printf("TrackingThreadBase(%p): failed to write capture to file %s\n", this, fnbuf);
      }

      // Reset capture index after writing one-shot
      m_captureFileIndex = 0;
    }

    // Derived class processing hook
    this->internalProcessOneCapture();
  }

  m_processingThreadAlive = false;
}
