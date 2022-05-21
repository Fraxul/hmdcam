#pragma once
#include <functional>
#include <map>
#include <queue>
#include <pthread.h>
#include <stdint.h>
#include "rhi/RHISurface.h"
#include <epoxy/egl.h>
#include "nvbufsurface.h"

class NvBuffer;
class NvVideoConverter;
class NvVideoEncoder;
struct v4l2_buffer;

class NvEncSession {
public:
  NvEncSession();
  ~NvEncSession();

  void setDimensions(uint32_t width, uint32_t height) { m_width = width; m_height = height; }
  void setBitrate(uint32_t bitsPerSecond) { m_bitsPerSecond = bitsPerSecond; }
  uint32_t bitrate() const { return m_bitsPerSecond; }
  void setFramerate(uint32_t numerator, uint32_t denominator) { m_framerateNumerator = numerator; m_framerateDenominator = denominator; }

  enum InputFormat {
    kInputFormatNV12,
    kInputFormatRGBX8,
  };

  void setInputFormat(InputFormat inputFormat) { m_inputFormat = inputFormat; }
  InputFormat inputFormat() const { return m_inputFormat; }

  size_t registerEncodedFrameDeliveryCallback(const std::function<void(const char*, size_t, struct timeval&)>& cb);
  void unregisterEncodedFrameDeliveryCallback(size_t cbId);

  void setUseGPUFrameSubmission(bool value);
  bool usingGPUFrameSubmission() const { return m_usingGPUFrameSubmission; }

  // CPU frame submission
  bool submitFrame(char* data, size_t length, bool blockIfQueueFull);

  // GPU frame submission
  RHISurface::ptr acquireSurface();
  bool submitSurface(RHISurface::ptr, bool blockIfQueueFull = false);

  bool isRunning() const { return m_startCount > 0; }

  void start();
  void stop();

protected:
  uint32_t m_width = 0, m_height = 0;
  uint32_t m_bitsPerSecond = 40000000;
  uint32_t m_framerateNumerator = 30, m_framerateDenominator = 1;
  uint32_t m_encoderPixfmt = 0;
  InputFormat m_inputFormat = kInputFormatNV12;
  bool m_usingGPUFrameSubmission = false;

  std::map<size_t, std::function<void(const char*, size_t, struct timeval&)> > m_encodedFrameDeliveryCallbacks;
  size_t m_encodedFrameDeliveryCallbackIdGen = 0;

  uint32_t m_startCount = 0;
  bool m_inShutdown = false;

  pthread_mutex_t m_stateLock;
  pthread_mutex_t m_callbackLock;

  NvVideoEncoder* m_enc = NULL;
  std::vector<NvBufSurface*> m_encOutputPlaneSurfaces;


  size_t m_currentSurfaceIndex = 0;
  std::vector<NvBufSurface*> m_vicInputSurfaces;
  std::vector<RHISurface::ptr> m_rhiSurfaces;
  std::vector<EGLImage> m_rhiSurfaceEGLImages;
  struct surfaceDmaBufInfo {
    surfaceDmaBufInfo() {
      for (int i = 0; i < NVBUF_MAX_PLANES; ++i) {
        plane_fds[i] = 0;
        plane_strides[i] = 0;
        plane_offsets[i] = 0;
      }
    }

    int fourcc = 0;
    int num_planes = 0;
    EGLuint64KHR modifiers = 0;

    int plane_fds[NVBUF_MAX_PLANES];
    EGLint plane_strides[NVBUF_MAX_PLANES];
    EGLint plane_offsets[NVBUF_MAX_PLANES];

    NvBufSurface* nvBuf = NULL;
  };
  std::vector<surfaceDmaBufInfo> m_rhiSurfaceDmaBufs;

  std::queue<std::pair<ssize_t, EGLSyncKHR> > m_gpuSubmissionQueue;
  pthread_mutex_t m_gpuSubmissionQueueLock;
  pthread_cond_t m_gpuSubmissionQueueCond;
  pthread_t m_cudaWorkerThread;

  std::queue<NvBuffer*> m_encoderOutputPlaneBufferQueue;

  static bool encoder_capture_plane_dq_callback_thunk(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer, void *arg);
  static void* cudaWorker_thunk(void*);

  bool encoder_capture_plane_dq_callback(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer);

  void cudaWorker();

private:
  // noncopyable
  NvEncSession(const NvEncSession&);
  NvEncSession& operator=(const NvEncSession&);
};

