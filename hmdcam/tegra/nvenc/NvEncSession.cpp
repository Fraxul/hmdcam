#include "NvEncSession.h"
#include "hmdcam/tegra/nvenc/NvEncSurfaceVK.h"
#include "rhi/RHI.h"
#include "rhi/RHISurface.h"
#include <cassert>
#include <linux/videodev2.h>
#include <linux/v4l2-controls.h>
#include <libv4l2.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/time.h>
#include <time.h>

#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "NvLogging.h"
#include "NvVideoEncoder.h"
#include "NvUtils.h"

#define die(msg, ...)                         \
  do {                                        \
    fprintf(stderr, msg "\n", ##__VA_ARGS__); \
    abort();                                  \
  } while (0)
#define CHECK_ZERO(x)                                              \
  if ((x) != 0) {                                                  \
    fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); \
    abort();                                                       \
  }
#define CHECK_TRUE(x)                                              \
  if (!(x)) {                                                      \
    fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); \
    abort();                                                       \
  }
#define CHECK_NOT_NULL(x)                                          \
  if ((x) == NULL) {                                               \
    fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); \
    abort();                                                       \
  }

static const uint32_t kInputBufferCount = 6;
static const uint32_t kOutputBufferCount = 10;

// Interval between IDR (keyframe) frames, in frames (~1 second at 30 fps). The
// stream needs recurring IDRs as random-access / recovery points: a decoder that
// joins mid-stream, drops a frame, or (like ffmpeg/ffplay) consumes the opening
// IDR while probing the container must be able to re-sync at the next one. The
// keyframe bitrate spike is bounded instead by the encoder's virtual buffer (see
// setVirtualBufferSize in start()).
static const uint32_t kKeyframeInterval = 30;

NvEncSession::NvEncSession(uint32_t _width, uint32_t _height) :
  m_width(_width),
  m_height(_height),
  m_encoderPixfmt(V4L2_PIX_FMT_H264) {

  // NvLogging
  // log_level = LOG_LEVEL_DEBUG;

  pthread_mutex_init(&m_stateLock, NULL);
  pthread_mutex_init(&m_callbackLock, NULL);

  pthread_mutex_init(&m_gpuSubmissionQueueLock, NULL);
  pthread_cond_init(&m_gpuSubmissionQueueCond, NULL);

  pthread_mutex_init(&m_submitWorkerActiveLock, NULL);

  // Allocate the rotating render-target pool. Each NvEncSurfaceVK owns an RGBA
  // NvBufSurface (the VIC input) imported into Vulkan as a color attachment, so
  // the render thread can draw directly into the surface the VIC will later
  // read -- no separate GL texture pool and no CUDA copy.
  m_surfaces.clear();
  m_currentSurfaceIndex = 0;
  m_surfaces.reserve(kInputBufferCount);
  for (size_t i = 0; i < kInputBufferCount; ++i) {
    m_surfaces.push_back(NvEncSurfaceVK::create(m_width, m_height));
  }

  // Create output plane DMABUFs
  {
    NvBufSurfaceAllocateParams encInputSurfaceAllocParams;
    memset(&encInputSurfaceAllocParams, 0, sizeof(encInputSurfaceAllocParams));
    encInputSurfaceAllocParams.params.width = m_width;
    encInputSurfaceAllocParams.params.height = m_height;
    encInputSurfaceAllocParams.params.layout = NVBUF_LAYOUT_PITCH;
    encInputSurfaceAllocParams.params.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
    encInputSurfaceAllocParams.params.memType = NVBUF_MEM_SURFACE_ARRAY;
    encInputSurfaceAllocParams.memtag = NvBufSurfaceTag_VIDEO_ENC;

    m_encOutputPlaneSurfaces.resize(kInputBufferCount);
    for (uint32_t i = 0; i < kInputBufferCount; i++) {
      CHECK_ZERO(NvBufSurfaceAllocate(&m_encOutputPlaneSurfaces[i], 1, &encInputSurfaceAllocParams));
      m_encOutputPlaneSurfaces[i]->numFilled = 1;
    }
  }
}

NvEncSession::~NvEncSession() {

  if (m_submitWorkerThreadRunning) {
    pthread_mutex_lock(&m_gpuSubmissionQueueLock);
    m_gpuSubmissionQueue.push(std::make_pair(static_cast<ssize_t>(-1), RHIFence::ptr()));
    pthread_cond_broadcast(&m_gpuSubmissionQueueCond);
    pthread_mutex_unlock(&m_gpuSubmissionQueueLock);
    m_submitWorkerThread.join();
    printf("NvEncSession: submit worker stopped\n");
    m_submitWorkerThreadRunning = false;
  }

  // Make sure that the encoder has been completely stopped before deleting anything
  pthread_mutex_lock(&m_stateLock);
  assert(m_startCount == 0);
  pthread_mutex_unlock(&m_stateLock);

  pthread_mutex_destroy(&m_stateLock);
  pthread_mutex_destroy(&m_callbackLock);

  pthread_mutex_destroy(&m_submitWorkerActiveLock);

  pthread_mutex_destroy(&m_gpuSubmissionQueueLock);
  pthread_cond_destroy(&m_gpuSubmissionQueueCond);

  // Releasing the NvEncSurfaceVK pool frees both the VK imports and the
  // underlying RGBA NvBufSurfaces.
  m_surfaces.clear();

  for (size_t i = 0; i < m_encOutputPlaneSurfaces.size(); ++i) {
    NvBufSurfaceDestroy(m_encOutputPlaneSurfaces[i]);
  }
  m_encOutputPlaneSurfaces.clear();
}

size_t NvEncSession::registerEncodedFrameDeliveryCallback(const std::function<void(const char*, size_t, struct timeval&)>& cb) {
  pthread_mutex_lock(&m_callbackLock);

  size_t cbid = ++m_encodedFrameDeliveryCallbackIdGen;
  m_encodedFrameDeliveryCallbacks[cbid] = cb;

  pthread_mutex_unlock(&m_callbackLock);
  return cbid;
}

void NvEncSession::unregisterEncodedFrameDeliveryCallback(size_t cbId) {
  pthread_mutex_lock(&m_callbackLock);
  m_encodedFrameDeliveryCallbacks.erase(cbId);
  pthread_mutex_unlock(&m_callbackLock);
}

RHISurface::ptr NvEncSession::acquireSurface() {
  // Prevents stalling the main render thread during encoder startup, when m_stateLock may be held for a long time.
  if (pthread_mutex_trylock(&m_stateLock) != 0)
    return RHISurface::ptr();

  RHISurface::ptr res;
  // Only return a surface if the encoder is running.
  if (m_startCount && !m_surfaces.empty())
    res = m_surfaces[m_currentSurfaceIndex];

  pthread_mutex_unlock(&m_stateLock);
  return res;
}

bool NvEncSession::submitSurface(RHISurface::ptr surface, bool blockIfQueueFull) {
  if (pthread_mutex_trylock(&m_stateLock) != 0)
    return false;

  // If the encoder is not running or is in shutdown, just ignore the submission.
  if (m_startCount == 0 || m_inShutdown) {
    pthread_mutex_unlock(&m_stateLock);
    return false;
  }

  assert(m_surfaces[m_currentSurfaceIndex].get() == surface.get());

  bool res = true;

  pthread_mutex_lock(&m_gpuSubmissionQueueLock);
  if (m_gpuSubmissionQueue.size() >= m_surfaces.size()) {
    printf("NvEncSession::submitSurface: queue is full\n");
    res = false; // queue is full
  } else {
    // Register a fence that signals when this frame's GPU work (including the
    // render pass that just filled this surface) completes. The worker waits
    // on it before handing the surface to the VIC. Called on the render thread,
    // which owns the RHI frame state. Replaces the old EGLSync-from-GL path.
    RHIFence::ptr fence = rhi()->registerFrameCompletionFence();
    m_gpuSubmissionQueue.push(std::make_pair(static_cast<ssize_t>(m_currentSurfaceIndex), fence));
    m_currentSurfaceIndex++;
    if (m_currentSurfaceIndex >= m_surfaces.size())
      m_currentSurfaceIndex = 0;
  }
  pthread_cond_broadcast(&m_gpuSubmissionQueueCond);
  pthread_mutex_unlock(&m_gpuSubmissionQueueLock);

  pthread_mutex_unlock(&m_stateLock);
  return res;
}

void NvEncSession::submitWorker() {
  prctl(PR_SET_NAME, "NvEncSessn-submit", 0, 0, 0);

  // Orientation of the rendered RGBA surface relative to the encoder. The
  // GL-era path rendered with GL's bottom-left origin and flipped Y in the VIC
  // to hand the encoder a top-left-origin frame. The VK RHI renders with a
  // top-left-origin viewport (RHIVK::setViewport uses a positive-height
  // viewport, not the negative-height flip trick), so the surface is already
  // upright and no VIC flip is needed. If the on-device stream comes out
  // vertically mirrored, flip this to true.
  constexpr bool kFlipYForEncoder = false;

  // VIC session for the RGBA -> YUV420 conversion. Unlike the old path this
  // worker needs no EGL context and no CUDA context: it only waits on a
  // VkFence (host-side) and drives the VIC + V4L2.
  NvBufSurfTransformConfigParams config_params = {NvBufSurfTransformCompute_VIC, 0, NULL};
  CHECK_ZERO(NvBufSurfTransformSetSessionParams(&config_params));

  while (true) {
    // Wait for next surface index in gpu submission queue
    pthread_mutex_lock(&m_gpuSubmissionQueueLock);
    while (m_gpuSubmissionQueue.empty()) {
      pthread_cond_wait(&m_gpuSubmissionQueueCond, &m_gpuSubmissionQueueLock);
    }
    ssize_t surfaceIdx = m_gpuSubmissionQueue.front().first;
    RHIFence::ptr surfaceFence = m_gpuSubmissionQueue.front().second;
    size_t submissionQueueSize = m_gpuSubmissionQueue.size();
    m_gpuSubmissionQueue.pop();
    pthread_mutex_unlock(&m_gpuSubmissionQueueLock);

    if (surfaceIdx < 0) {
      fprintf(stderr, "NvEncSession::submitWorker(): thread stop requested\n");
      break;
    }
    pthread_mutex_lock(&m_submitWorkerActiveLock);

    NvBuffer* encoderInputBuffer = NULL;

    if (m_encoderOutputPlaneBufferQueue.empty()) {
      struct v4l2_buffer v4l2_buf;
      struct v4l2_plane planes[MAX_PLANES];
      NvBuffer* buffer;
      NvBuffer* shared_buffer;

      memset(&v4l2_buf, 0, sizeof(v4l2_buf));
      memset(planes, 0, sizeof(planes));
      v4l2_buf.m.planes = planes;

      if (m_enc->output_plane.dqBuffer(v4l2_buf, &buffer, &shared_buffer, -1) < 0) {
        die("Failed to dequeue buffer from encoder output plane");
      }

      encoderInputBuffer = m_enc->output_plane.getNthBuffer(v4l2_buf.index);
    } else {
      encoderInputBuffer = m_encoderOutputPlaneBufferQueue.front();
      m_encoderOutputPlaneBufferQueue.pop();
    }

    // Wait for the render thread's GPU work that filled this surface to finish.
    // The fence signals on frame-completion (see RHIVK::registerFrameCompletionFence).
    // A 1s timeout guards against a frame that never submitted (e.g. a stall on
    // shutdown); on timeout we skip the conversion and recycle the encoder
    // buffer rather than encode undefined contents.
    if (surfaceFence && !surfaceFence->wait(1'000'000'000ull /* 1s */)) {
      fprintf(stderr, "NvEncSession::submitWorker(): render fence wait timed out; dropping frame. submissionQueueSize=%zu\n", submissionQueueSize);
      m_encoderOutputPlaneBufferQueue.push(encoderInputBuffer);
      pthread_mutex_unlock(&m_submitWorkerActiveLock);
      continue;
    }

    // VIC: convert the rendered RGBA surface directly into the YUV420 encoder
    // input plane. The RGBA source is the same NvBufSurface the render thread
    // drew into -- no intervening copy.
    NvBufSurfTransformRect src_rect, dest_rect;
    src_rect.top = 0;
    src_rect.left = 0;
    src_rect.width = m_width;
    src_rect.height = m_height;
    dest_rect.top = 0;
    dest_rect.left = 0;
    dest_rect.width = m_width;
    dest_rect.height = m_height;

    NvBufSurfTransformParams xfParams;
    memset(&xfParams, 0, sizeof(xfParams));

    xfParams.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    if (kFlipYForEncoder) {
      xfParams.transform_flag |= NVBUFSURF_TRANSFORM_FLIP;
      xfParams.transform_flip = NvBufSurfTransform_FlipY;
    }
    xfParams.transform_filter = NvBufSurfTransformInter_Algo3;
    xfParams.src_rect = &src_rect;
    xfParams.dst_rect = &dest_rect;

    NvBufSurface* srcSurf = m_surfaces[surfaceIdx]->nvBufSurface();
    NvBufSurface* encoderInputSrf = m_encOutputPlaneSurfaces[encoderInputBuffer->index];
    NvBufSurfTransform_Error xfErr = NvBufSurfTransform(/*src=*/ srcSurf, /*dst=*/ encoderInputSrf, &xfParams);
    if (xfErr != NvBufSurfTransformError_Success) {
      switch (xfErr) {
        case NvBufSurfTransformError_ROI_Error: die("NvBufSurfTransformError_ROI_Error");
        case NvBufSurfTransformError_Invalid_Params: die("NvBufSurfTransformError_Invalid_Params");
        case NvBufSurfTransformError_Execution_Error: die("NvBufSurfTransformError_Execution_Error");
        case NvBufSurfTransformError_Unsupported: die("NvBufSurfTransformError_Unsupported");
        default: die("NvBufSurfTransform bad result %d", xfErr);
      }
    }


    // V4L2 handoff
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, sizeof(planes));

    v4l2_buf.index = encoderInputBuffer->index;
    v4l2_buf.m.planes = planes;
    gettimeofday(&v4l2_buf.timestamp, NULL);

    // bytesused gets reset when the buffer is dequeued, so we have to re-specify it every time before qBuffer
    for (uint32_t planeIdx = 0; planeIdx < encoderInputBuffer->n_planes; ++planeIdx) {
      encoderInputBuffer->planes[planeIdx].fd = m_encOutputPlaneSurfaces[encoderInputBuffer->index]->surfaceList[0].bufferDesc;
      encoderInputBuffer->planes[planeIdx].bytesused = m_encOutputPlaneSurfaces[encoderInputBuffer->index]->surfaceList[0].planeParams.psize[planeIdx];
    }

    int ret = m_enc->output_plane.qBuffer(v4l2_buf, encoderInputBuffer);
    if (ret < 0)
      die("Error while queueing buffer at encoder output plane");
    pthread_mutex_unlock(&m_submitWorkerActiveLock);
  }
}

void NvEncSession::start() {
  int ret;

  assert(m_width && m_height);
  assert(m_framerateDenominator && m_framerateNumerator);
  assert(m_bitsPerSecond);

  pthread_mutex_lock(&m_stateLock);

  ++m_startCount;
  if (m_startCount > 1) {
    // Was already running, just increase the refcount.
    pthread_mutex_unlock(&m_stateLock);
    return;
  }
  printf("NvEncSession: starting\n");
  m_inShutdown = false;

  // Reset the strictly-increasing emission-timestamp guard for this stream.
  m_lastEmitTimeUs = 0;

  m_enc = NvVideoEncoder::createVideoEncoder("enc0");
  if (!m_enc) die("Could not create encoder");

  // It is necessary that Capture Plane format be set before Output Plane format.
  // Set encoder capture plane format. It is necessary to set width and height on the capture plane as well.
  //
  // The last argument is the per-buffer sizeimage for *encoded* output: it must
  // be large enough to hold the single largest access unit the encoder emits, or
  // the V4L2 encoder truncates the frame to fit (bytesused is capped at capacity)
  // and the decoder sees a slice cut off mid-macroblock -- a clean top, then green
  // to the bottom. With B-frames disabled and no slice splitting, each frame is one
  // slice NAL, and a 4K IDR can run well past the 2 MiB the NVIDIA samples use for
  // 1080p. Size it to one byte per pixel (~7.9 MiB at 3840x2160), which comfortably
  // covers even a quality-uncapped first IDR while scaling with resolution. The
  // remote-debug stream (RenderDebug.cpp) writes whatever NAL the encoder emits
  // straight to the client socket, so this buffer is the only size cap in the path.
  const uint32_t encodedFrameBufferSize = m_width * m_height;
  ret = m_enc->setCapturePlaneFormat(m_encoderPixfmt, m_width, m_height, encodedFrameBufferSize);
  if (ret < 0) die("Could not set output plane format");

  ret = m_enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, m_width, m_height);
  if (ret < 0) die("Could not set output plane format");

  ret = m_enc->setHWPresetType(V4L2_ENC_HW_PRESET_MEDIUM);
  if (ret < 0) die("Could not set encoder hardware quality preset");

  ret = m_enc->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_VBR);
  if (ret < 0) die("Could not set rate control mode");

  ret = m_enc->setBitrate(m_bitsPerSecond / 2);
  if (ret < 0) die("Could not set bitrate");

  ret = m_enc->setPeakBitrate(m_bitsPerSecond);
  if (ret < 0) die("Could not set bitrate");

  if (m_encoderPixfmt == V4L2_PIX_FMT_H264) {
    ret = m_enc->setProfile(V4L2_MPEG_VIDEO_H264_PROFILE_HIGH);
  } else {
    ret = m_enc->setProfile(V4L2_MPEG_VIDEO_H265_PROFILE_MAIN);
  }
  if (ret < 0) die("Could not set encoder profile");

  if (m_encoderPixfmt == V4L2_PIX_FMT_H264) {
    ret = m_enc->setLevel(V4L2_MPEG_VIDEO_H264_LEVEL_5_0);
    if (ret < 0) die("Could not set encoder level");
  }

  ret = m_enc->setFrameRate(m_framerateNumerator, m_framerateDenominator);
  if (ret < 0) die("Could not set framerate");

  // Streaming config tuned for low-latency live delivery to a single client:
  //  - SPS/PPS inline at every IDR so a decoder can initialize at any keyframe
  //    (mid-stream attach, post-probe start, or recovery after a dropped frame).
  //  - Recurring IDRs as random-access points -- see kKeyframeInterval. We tried
  //    periodic intra-refresh with no recurring IDR to avoid keyframe bitrate
  //    spikes, but a stream with a single IDR cannot be re-synced once that IDR is
  //    gone: ffmpeg/ffplay consumes it while probing the container and then has no
  //    entry point, so it never displays. Intra-refresh's payoff (recovery without
  //    an IDR) needs a recovery-point SEI the encoder does not emit, and is only
  //    worthwhile on lossy links -- not on this lossless TCP path.
  //  - A virtual-buffer (VBV/HRD) cap so the keyframe does not burst the network:
  //    the encoder spends extra QP on the IDR to stay within the buffer instead of
  //    spiking the instantaneous bitrate. Sized (in bits) at ~0.5 s of the
  //    configured bitrate; lower it to cap spikes more tightly, at some
  //    keyframe-quality cost.
  m_enc->setInsertSpsPpsAtIdrEnabled(true);
  m_enc->setIFrameInterval(kKeyframeInterval);
  m_enc->setIDRInterval(kKeyframeInterval);
  m_enc->setVirtualBufferSize(m_bitsPerSecond / 2);
  m_enc->setMaxPerfMode(1);
  m_enc->setNumBFrames(0); // Disable B-frames for low latency
  // Insert VUI so that the RTSP server can pull framerate information out of it
  //m_enc->setInsertVuiEnabled(true);

  // REQBUF on encoder output plane buffers
  // DMABUF is used here since it is a shared buffer allocated by another component
  // setupPlane can't handle creating the DMABUFs, so we'll do that later.
  ret = m_enc->output_plane.setupPlane(V4L2_MEMORY_DMABUF, kInputBufferCount, false, false);
  if (ret < 0) die("Could not setup encoder output plane");

  // Query, Export and Map the output plane buffers so that we can write
  // encoded data from the buffers
  ret = m_enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, kOutputBufferCount, true, false);
  if (ret < 0) die("Could not setup encoder capture plane");

  // output plane STREAMON
  ret = m_enc->output_plane.setStreamStatus(true);
  if (ret < 0) die("Error in encoder output plane streamon");

  // capture plane STREAMON
  ret = m_enc->capture_plane.setStreamStatus(true);
  if (ret < 0) die("Error in encoder capture plane streamon");

  // startDQThread starts a thread internally which calls the
  // encoder_capture_plane_dq_callback whenever a buffer is dequeued
  // on the plane
  m_enc->capture_plane.setDQThreadCallback(encoder_capture_plane_dq_callback_thunk);
  m_enc->capture_plane.startDQThread(this);

  // Add empty encoder output plane buffers to m_encoderOutputPlaneBufferQueue
  assert(m_enc->output_plane.getNumBuffers() == kInputBufferCount);
  for (uint32_t i = 0; i < m_enc->output_plane.getNumBuffers(); i++) {
    NvBuffer* buf = m_enc->output_plane.getNthBuffer(i);
    m_encoderOutputPlaneBufferQueue.push(buf);
  }

  // Enqueue all the empty encoder capture plane buffers
  for (uint32_t i = 0; i < m_enc->capture_plane.getNumBuffers(); i++) {
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

    v4l2_buf.index = i;
    v4l2_buf.m.planes = planes;

    ret = m_enc->capture_plane.qBuffer(v4l2_buf, NULL);
    if (ret < 0) die("Error while queueing buffer at capture plane");
  }

  // Start the submit worker thread
  while (!m_gpuSubmissionQueue.empty())
    m_gpuSubmissionQueue.pop();

  if (!m_submitWorkerThreadRunning) {
    m_submitWorkerThread = FxThread(&NvEncSession::submitWorker, this);
    m_submitWorkerThreadRunning = true;
  }

  printf("NvEncSession: started.\n");
  pthread_mutex_unlock(&m_stateLock);
}

void NvEncSession::stop() {
  pthread_mutex_lock(&m_stateLock);

  --m_startCount;
  if (m_startCount > 0) {
    // Still running
    pthread_mutex_unlock(&m_stateLock);
    return;
  }

  printf("NvEncSession: stopping\n");

  m_inShutdown = true;

  // Ensure the submit worker thread has drained its work queue
  pthread_mutex_lock(&m_gpuSubmissionQueueLock);
  pthread_mutex_lock(&m_submitWorkerActiveLock); // Wait for the submit worker to finish and return to servicing the submission queue
  while (!m_gpuSubmissionQueue.empty()) {
    m_gpuSubmissionQueue.pop();
  }
  pthread_mutex_unlock(&m_submitWorkerActiveLock);
  pthread_mutex_unlock(&m_gpuSubmissionQueueLock);

  // Wait till capture plane DQ Thread finishes
  // i.e. all the capture plane buffers are dequeued
  m_enc->capture_plane.waitForDQThread(2000);

  // Shut down the encoder
  m_enc->abort();

  delete m_enc;
  m_enc = NULL;
  while (!m_encoderOutputPlaneBufferQueue.empty()) m_encoderOutputPlaneBufferQueue.pop();

  printf("NvEncSession: stopped.\n");
  pthread_mutex_unlock(&m_stateLock);
}

bool NvEncSession::encoder_capture_plane_dq_callback(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer) {

  if (m_inShutdown)
    return false; // cancel operations

  if (!v4l2_buf) {
    die("Failed to dequeue buffer from encoder capture plane");
    return false;
  }

  if (buffer->planes[0].bytesused >= 4) {
    // Presentation timestamp for this access unit. This is a *live* stream, so we
    // stamp each frame with the real time it is emitted (CLOCK_MONOTONIC) rather
    // than an evenly-spaced frame-counter cadence. An assumed-even cadence drifts
    // against the true frame-production rate -- the render loop cannot deliver
    // exactly the nominal fps -- and the downstream MPEG-TS muxer's PCR is slaved
    // to this value, so an even cadence makes the player's clock run slightly fast,
    // its decode buffer underruns, and it periodically declares pictures late and
    // re-buffers. A real emission timestamp keeps PCR matched to actual delivery.
    //
    // The DQ thread can emit two access units within the same microsecond during
    // pipeline catch-up, so clamp to keep the delivered timestamp strictly
    // increasing (a stalled or backward PTS/PCR corrupts playback timing).
    struct timeval pts;
    {
      struct timespec now;
      clock_gettime(CLOCK_MONOTONIC, &now);
      uint64_t nowUs = (static_cast<uint64_t>(now.tv_sec) * 1000000ull) + (now.tv_nsec / 1000ull);
      if (nowUs <= m_lastEmitTimeUs)
        nowUs = m_lastEmitTimeUs + 1;
      m_lastEmitTimeUs = nowUs;

      pts.tv_sec = static_cast<time_t>(nowUs / 1000000ull);
      pts.tv_usec = static_cast<suseconds_t>(nowUs % 1000000ull);
    }

    pthread_mutex_lock(&m_callbackLock);

    const char* data = reinterpret_cast<const char*>(buffer->planes[0].data);
    size_t dataSize = buffer->planes[0].bytesused;

    // Deliver the whole access unit (one encoded frame) to each consumer. The
    // encoder hands us exactly one access unit per capture-plane buffer -- B-frames
    // are disabled and there is one slice per frame, with SPS/PPS inline only at
    // IDR -- already in Annex-B byte-stream form (start-code delimited NALs).
    for (const auto& cbIt : m_encodedFrameDeliveryCallbacks) {
      cbIt.second(data, dataSize, pts);
    }

    pthread_mutex_unlock(&m_callbackLock);
  }

  if (m_enc->capture_plane.qBuffer(*v4l2_buf, NULL) < 0) {
    die("Error while Qing buffer at capture plane");
    return false;
  }

  return true;
}

/*static*/ bool NvEncSession::encoder_capture_plane_dq_callback_thunk(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer, void* arg) {
  return reinterpret_cast<NvEncSession*>(arg)->encoder_capture_plane_dq_callback(v4l2_buf, buffer, shared_buffer);
}
