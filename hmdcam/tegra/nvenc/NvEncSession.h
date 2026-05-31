#pragma once
#include <functional>
#include <map>
#include <queue>
#include <vector>
#include <pthread.h>
#include <stdint.h>
#include <boost/smart_ptr/intrusive_ptr.hpp>
#include "rhi/RHISurface.h"
#include "rhi/RHIFence.h"
#include "nvbufsurface.h"

class NvBuffer;
class NvVideoEncoder;
struct v4l2_buffer;
// Forward-declared to keep vulkan-hpp out of the RTSP/live555 translation units
// that include this header; NvEncSession.cpp includes the full definition.
class NvEncSurfaceVK;

// NvEncSession: drives the Tegra H.264 encoder for the remote-debug RTSP
// stream. The render thread acquires a surface, draws into it via the RHI, and
// submits it; a worker thread converts and feeds it to the encoder.
//
// Post-Vulkan-migration data path (no CUDA / no EGLImage):
//   1. acquireSurface() hands out an NvEncSurfaceVK from a rotating pool. Each
//      wraps an RGBA NvBufSurface imported into Vulkan as a color attachment.
//   2. The render thread draws directly into that surface (RHI render pass).
//   3. submitSurface() registers an RHIFence (signals on frame-GPU-completion)
//      and queues the (surface index, fence) pair for the worker.
//   4. The worker waits the fence, then NvBufSurfTransform (VIC) converts the
//      RGBA surface to the YUV420 encoder input plane and hands it to V4L2.
class NvEncSession {
public:
  NvEncSession(uint32_t width, uint32_t height);
  ~NvEncSession();

  void setBitrate(uint32_t bitsPerSecond) { m_bitsPerSecond = bitsPerSecond; }
  uint32_t bitrate() const { return m_bitsPerSecond; }
  void setFramerate(uint32_t numerator, uint32_t denominator) {
    m_framerateNumerator = numerator;
    m_framerateDenominator = denominator;
  }

  size_t registerEncodedFrameDeliveryCallback(const std::function<void(const char*, size_t, struct timeval&)>& cb);
  void unregisterEncodedFrameDeliveryCallback(size_t cbId);

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

  std::map<size_t, std::function<void(const char*, size_t, struct timeval&)>> m_encodedFrameDeliveryCallbacks;
  size_t m_encodedFrameDeliveryCallbackIdGen = 0;

  uint32_t m_startCount = 0;
  bool m_inShutdown = false;

  pthread_mutex_t m_stateLock;
  pthread_mutex_t m_callbackLock;

  NvVideoEncoder* m_enc = NULL;
  std::vector<NvBufSurface*> m_encOutputPlaneSurfaces;

  // Rotating pool of render-target surfaces (RGBA NvBufSurface + VK import).
  size_t m_currentSurfaceIndex = 0;
  std::vector<boost::intrusive_ptr<NvEncSurfaceVK>> m_surfaces;

  // (surface index, render-completion fence) pairs awaiting VIC conversion.
  std::queue<std::pair<ssize_t, RHIFence::ptr>> m_gpuSubmissionQueue;
  pthread_mutex_t m_gpuSubmissionQueueLock;
  pthread_cond_t m_gpuSubmissionQueueCond;
  pthread_t m_submitWorkerThread;
  pthread_mutex_t m_submitWorkerActiveLock;
  bool m_submitWorkerThreadRunning = false;

  std::queue<NvBuffer*> m_encoderOutputPlaneBufferQueue;

  static bool encoder_capture_plane_dq_callback_thunk(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer, void* arg);
  static void* submitWorker_thunk(void*);

  bool encoder_capture_plane_dq_callback(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer);

  void submitWorker();

private:
  // noncopyable
  NvEncSession(const NvEncSession&);
  NvEncSession& operator=(const NvEncSession&);
};
