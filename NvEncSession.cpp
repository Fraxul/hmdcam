#include "NvEncSession.h"
#include <cassert>
#include <linux/videodev2.h>
#include <linux/v4l2-controls.h>
#include <stdio.h>
#include <string.h>

#include "nvbuf_utils.h"
#include "NvVideoEncoder.h"
#include "NvVideoConverter.h"
#include "NvUtils.h"

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

NvEncSession::NvEncSession() : m_width(0), m_height(0), m_bitsPerSecond(8000000),
  m_framerateNumerator(30), m_framerateDenominator(1),
  m_encoderPixfmt(V4L2_PIX_FMT_H264),
  m_inputFormat(kInputFormatNV12),
  m_encodedFrameDeliveryCallbackIdGen(0),
  m_startCount(0) {

  pthread_mutex_init(&m_stateLock, NULL);
  pthread_mutex_init(&m_callbackLock, NULL);

  pthread_mutex_init(&m_encoderOutputPlaneBufferQueueLock, NULL);
  pthread_cond_init(&m_encoderOutputPlaneBufferQueueCond, NULL);
  pthread_mutex_init(&m_conv0OutputPlaneBufferQueueLock, NULL);
  pthread_cond_init(&m_conv0OutputPlaneBufferQueueCond, NULL);
}

NvEncSession::~NvEncSession() {
  // Make sure that the encoder has been completely stopped before deleting anything
  pthread_mutex_lock(&m_stateLock);
  assert(m_startCount == 0);
  pthread_mutex_unlock(&m_stateLock);

  pthread_mutex_destroy(&m_stateLock);
  pthread_mutex_destroy(&m_callbackLock);

  pthread_mutex_destroy(&m_encoderOutputPlaneBufferQueueLock);
  pthread_cond_destroy(&m_encoderOutputPlaneBufferQueueCond);
  pthread_mutex_destroy(&m_conv0OutputPlaneBufferQueueLock);
  pthread_cond_destroy(&m_conv0OutputPlaneBufferQueueCond);

}

size_t NvEncSession::registerEncodedFrameDeliveryCallback(const std::function<void(const char*, size_t)>& cb) {
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

bool NvEncSession::submitFrame(char* data, size_t length, bool blockIfQueueFull) {

  if (m_inShutdown && length != 0)
    return false; // Reject frames submitted during shutdown

  if (!((length == 0 && m_inShutdown) || length == inputFrameSize())) {
    printf("NvEncSession::submitFrame(): short frame submitted, length %zu (require %zu)\n", length, inputFrameSize());
    return false;
  }

  pthread_mutex_lock(&m_conv0OutputPlaneBufferQueueLock);
  while (m_conv0OutputPlaneBufferQueue.empty()) {
    if (!blockIfQueueFull) {
      pthread_mutex_unlock(&m_conv0OutputPlaneBufferQueueLock);
      return false; // frame dropped
    }

    pthread_cond_wait(&m_conv0OutputPlaneBufferQueueCond, &m_conv0OutputPlaneBufferQueueLock);
  }

  NvBuffer *buffer = m_conv0OutputPlaneBufferQueue.front();
  m_conv0OutputPlaneBufferQueue.pop();
  pthread_mutex_unlock(&m_conv0OutputPlaneBufferQueueLock);

  //printf("NvEncSession::submitFrame() data=%p length=%zu blockIfQueueFull=%u buffer->index=%u\n", data, length, blockIfQueueFull, buffer->index);

  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(planes, 0, sizeof(planes));

  v4l2_buf.index = buffer->index;
  v4l2_buf.m.planes = planes;

  if (data && length) {
    const char* readPtr = data; 
    for (size_t planeIdx = 0; planeIdx < buffer->n_planes; ++planeIdx) {
      NvBuffer::NvBufferPlane &plane = buffer->planes[planeIdx];
      // TODO: this assumes input data is tightly packed.
      size_t bytesPerRow = plane.fmt.bytesperpixel * plane.fmt.width;
      plane.bytesused = plane.fmt.stride * plane.fmt.height;

      char* writePtr = (char *) plane.data;
      for (size_t y = 0; y < plane.fmt.height; ++y) {
        memcpy(writePtr, readPtr, bytesPerRow);
        readPtr += bytesPerRow;
        writePtr += plane.fmt.stride;
      }
    }
  } else {
    for (size_t planeIdx = 0; planeIdx < buffer->n_planes; ++planeIdx) {
      buffer->planes[planeIdx].bytesused = 0;
    }
  }

  int ret = m_conv0->output_plane.qBuffer(v4l2_buf, NULL);
  if (ret < 0)
    die("Error while queueing buffer at output plane");

  return true;
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

  m_enc = NvVideoEncoder::createVideoEncoder("enc0");
  if (!m_enc) die("Could not create encoder");

  m_conv0 = NvVideoConverter::createVideoConverter("conv0");
  if (!m_conv0) die("Could not create Video Converter");

  // TODO parameterize input format
  uint32_t outputPixfmt = 0;
  switch (m_inputFormat) {
    // V4L2 format enums are in linux/videodev2.h
    case kInputFormatNV12: outputPixfmt = V4L2_PIX_FMT_NV12M; break;
    case kInputFormatRGBX8: outputPixfmt = V4L2_PIX_FMT_ABGR32; break;
    default: die("Invalid input format enum %u", m_inputFormat);
  };

  ret = m_conv0->setOutputPlaneFormat(outputPixfmt, m_width, m_height, V4L2_NV_BUFFER_LAYOUT_PITCH);
  if (ret < 0) die("Could not set output plane format for conv0");

  ret = m_conv0->setCapturePlaneFormat(V4L2_PIX_FMT_YUV420M, m_width, m_height, V4L2_NV_BUFFER_LAYOUT_BLOCKLINEAR);
  if (ret < 0) die("Could not set capture plane format for conv0");

  // XXX: flip the video top to bottom for testing, to prove that the video converter component works
  ret = m_conv0->setFlipMethod(V4L2_FLIP_METHOD_VERT);
  if (ret < 0) die("Could not set flip method");

  // It is necessary that Capture Plane format be set before Output Plane format.
  // Set encoder capture plane format. It is necessary to set width and height on the capture plane as well.
  ret = m_enc->setCapturePlaneFormat(m_encoderPixfmt, m_width, m_height, 2 * 1024 * 1024);
  if (ret < 0) die("Could not set output plane format");

  ret = m_enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, m_width, m_height);
  if (ret < 0) die("Could not set output plane format");

  ret = m_enc->setBitrate(m_bitsPerSecond);
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

  // Set up for streaming -- insert SPS and PPS every 60 frames so the decoder can sync to an in-progress stream.
  m_enc->setInsertSpsPpsAtIdrEnabled(true);
  m_enc->setIDRInterval(60 /*frames*/);

  // REQBUF, EXPORT and MAP conv0 output plane buffers
  ret = m_conv0->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 10, false, true);
  if (ret < 0) die("Error while setting up output plane for conv0");

  // REQBUF and EXPORT conv0 capture plane buffers
  // No need to MAP since buffer will be shared to next component
  // and not read in application
  ret = m_conv0->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 10, false, false);
  if (ret < 0) die("Error while setting up capture plane for conv0");

  // conv0 output plane STREAMON
  ret = m_conv0->output_plane.setStreamStatus(true);
  if (ret < 0) die("Error in output plane streamon for conv0");

  // conv0 capture plane STREAMON
  ret = m_conv0->capture_plane.setStreamStatus(true);
  if (ret < 0) die("Error in capture plane streamon for conv0");

  // REQBUF on encoder output plane buffers
  // DMABUF is used here since it is a shared buffer allocated by another component
  ret = m_enc->output_plane.setupPlane(V4L2_MEMORY_DMABUF, 10, false, false);
  if (ret < 0) die("Could not setup encoder output plane");

  // Query, Export and Map the output plane buffers so that we can write
  // encoded data from the buffers
  ret = m_enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 6, true, false);
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
  m_conv0->capture_plane.setDQThreadCallback(conv0_capture_dqbuf_thread_callback_thunk);
  m_conv0->capture_plane.startDQThread(this);
  m_enc->capture_plane.setDQThreadCallback(encoder_capture_plane_dq_callback_thunk);
  m_enc->capture_plane.startDQThread(this);

  // Enqueue all empty conv0 capture plane buffers
  for (uint32_t i = 0; i < m_conv0->capture_plane.getNumBuffers(); i++) {
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

    v4l2_buf.index = i;
    v4l2_buf.m.planes = planes;

    ret = m_conv0->capture_plane.qBuffer(v4l2_buf, NULL);
    if (ret < 0) die("Error while queueing buffer at conv0 capture plane");
  }

  // Add all empty encoder output plane buffers to m_encoderOutputPlaneBufferQueue
  for (uint32_t i = 0; i < m_enc->output_plane.getNumBuffers(); i++) {
    m_encoderOutputPlaneBufferQueue.push(m_enc->output_plane.getNthBuffer(i));
  }

  // Add all empty conv0 output plane buffers to m_conv0OutputPlaneBufferQueue
  for (uint32_t i = 0; i < m_conv0->output_plane.getNumBuffers(); i++) {
    m_conv0OutputPlaneBufferQueue.push(m_conv0->output_plane.getNthBuffer(i));
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

  // Compute the input frame size from a conv0 output plane buffer
  {
    NvBuffer* buffer = m_conv0->output_plane.getNthBuffer(0);
    m_inputFrameSize = 0;

    printf("NvEncSession::start(): Input frame planes:\n");
    for (size_t planeIdx = 0; planeIdx < buffer->n_planes; ++planeIdx) {
      NvBuffer::NvBufferPlane &plane = buffer->planes[planeIdx];
      printf("  %zu: %u x %u, %u bytes/pixel, %u bytes\n", planeIdx, plane.fmt.width, plane.fmt.height, plane.fmt.bytesperpixel, plane.fmt.bytesperpixel * plane.fmt.width * plane.fmt.height);

      // TODO: this assumes input data is tightly packed.
      size_t bytesPerRow = plane.fmt.bytesperpixel * plane.fmt.width;
      m_inputFrameSize += bytesPerRow * plane.fmt.height;
    }
    printf("NvEncSession::start(): Input frame %zu bytes total\n", m_inputFrameSize);
  }

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

  // Submit EOS
  submitFrame(NULL, 0, true);

  // Wait till capture plane DQ Thread finishes
  // i.e. all the capture plane buffers are dequeued
  m_conv0->waitForIdle(2000);
  m_enc->capture_plane.waitForDQThread(2000);
  delete m_conv0;
  delete m_enc;

  m_conv0 = NULL;
  m_enc = NULL;
  while (!m_encoderOutputPlaneBufferQueue.empty()) m_encoderOutputPlaneBufferQueue.pop();
  while (!m_conv0OutputPlaneBufferQueue.empty()) m_conv0OutputPlaneBufferQueue.pop();

  pthread_mutex_unlock(&m_stateLock);
}

bool NvEncSession::conv0_output_plane_dq() {
  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];
  NvBuffer *buffer;

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(planes, 0, sizeof(planes));
  v4l2_buf.m.planes = planes;
  v4l2_buf.length = m_enc->output_plane.getNumPlanes();

  if (m_conv0->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1) < 0) {
    die("Failed to dequeue buffer from conv0 output plane");
    return false;
  }

  // Add the dequeued buffer to conv0 empty output buffers queue
  pthread_mutex_lock(&m_conv0OutputPlaneBufferQueueLock);

  m_conv0OutputPlaneBufferQueue.push(buffer);
  pthread_cond_broadcast(&m_conv0OutputPlaneBufferQueueCond);
  pthread_mutex_unlock(&m_conv0OutputPlaneBufferQueueLock);

  return true;
}

bool NvEncSession::conv0_capture_dqbuf_thread_callback(struct v4l2_buffer *v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer) {
  NvBuffer *enc_buffer;
  struct v4l2_buffer enc_qbuf;
  struct v4l2_plane planes[MAX_PLANES];

  if (!v4l2_buf) {
    die("Failed to dequeue buffer from conv0 capture plane");
    return false;
  }

  // Get an empty enc output plane buffer from m_encoderOutputPlaneBufferQueue
  pthread_mutex_lock(&m_encoderOutputPlaneBufferQueueLock);
  while (m_encoderOutputPlaneBufferQueue.empty()) {
    pthread_cond_wait(&m_encoderOutputPlaneBufferQueueCond, &m_encoderOutputPlaneBufferQueueLock);
  }

  enc_buffer = m_encoderOutputPlaneBufferQueue.front();
  m_encoderOutputPlaneBufferQueue.pop();
  pthread_mutex_unlock(&m_encoderOutputPlaneBufferQueueLock);

  memset(&enc_qbuf, 0, sizeof(enc_qbuf));
  memset(&planes, 0, sizeof(planes));

  enc_qbuf.index = enc_buffer->index;
  enc_qbuf.m.planes = planes;

  // A reference to buffer is saved which can be used when
  // buffer is dequeued from enc output plane
  if (m_enc->output_plane.qBuffer(enc_qbuf, buffer) < 0) {
    die("Error queueing buffer on encoder output plane");
    return false;
  }

  if (v4l2_buf->m.planes[0].bytesused == 0) {
    if (m_inShutdown) {
      return false;
    } else {
      printf("conv0_capture_dqbuf_thread_callback: spurious EOS\n");
    }
  }

  // If we're not at EOS, also handle dequeueing buffers from the output plane.
  conv0_output_plane_dq();

  return true;
}

bool NvEncSession::encoder_output_plane_dq() {
  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];
  NvBuffer *buffer;
  NvBuffer *shared_buffer;

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(planes, 0, sizeof(planes));
  v4l2_buf.m.planes = planes;
  v4l2_buf.length = m_enc->output_plane.getNumPlanes();

  if (m_enc->output_plane.dqBuffer(v4l2_buf, &buffer, &shared_buffer, -1) < 0) {
    die("Failed to dequeue buffer from encoder output plane");
    return false;
  }

  struct v4l2_buffer conv0_ret_qbuf;

  memset(&conv0_ret_qbuf, 0, sizeof(conv0_ret_qbuf));
  memset(&planes, 0, sizeof(planes));

  // Get the index of the conv0 capture plane shared buffer
  conv0_ret_qbuf.index = shared_buffer->index;
  conv0_ret_qbuf.m.planes = planes;

  // Add the dequeued buffer to encoder empty output buffers queue
  // queue the shared buffer back in conv0 capture plane
  pthread_mutex_lock(&m_encoderOutputPlaneBufferQueueLock);
  if (m_conv0->capture_plane.qBuffer(conv0_ret_qbuf, NULL) < 0) {
    die("Error queueing buffer on conv0 capture plane");
    return false;
  }
  m_encoderOutputPlaneBufferQueue.push(buffer);
  pthread_cond_broadcast(&m_encoderOutputPlaneBufferQueueCond);
  pthread_mutex_unlock(&m_encoderOutputPlaneBufferQueueLock);

  return true;
}

bool NvEncSession::encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer) {

  if (!v4l2_buf) {
    die("Failed to dequeue buffer from encoder capture plane");
    return false;
  }

  //write_encoder_output_frame(ctx->out_file, buffer);
  //ctx->bufferRingSource->asyncDeliverFrame(ctx->scheduler, (char*) buffer->planes[0].data, buffer->planes[0].bytesused);
  pthread_mutex_lock(&m_callbackLock);
  for (const auto& cbIt : m_encodedFrameDeliveryCallbacks) {
    cbIt.second((const char*) buffer->planes[0].data, buffer->planes[0].bytesused);
  }
  pthread_mutex_unlock(&m_callbackLock);

  if (m_enc->capture_plane.qBuffer(*v4l2_buf, NULL) < 0) {
    die("Error while Qing buffer at capture plane");
    return false;
  }

  // GOT EOS from encoder. Stop dqthread.
  if (buffer->planes[0].bytesused == 0) {
    if (m_inShutdown)
      return false;
    else
      printf("encoder_capture_plane_dq_callback: spurious EOS\n");
  }

  // Handle dq for output plane here as well. The component seems to lock up (dqBuffer hangs in v4l ioctl)
  // once EOS has been reached, so we only dq on the output plane if we haven't gotten EOS on the capture plane.
  // (The EOS buffer does make it through the component correctly -- only the dq stops working)
  return encoder_output_plane_dq();
}

/*static*/ bool NvEncSession::conv0_capture_dqbuf_thread_callback_thunk(struct v4l2_buffer *v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer, void* arg) {
  return reinterpret_cast<NvEncSession*>(arg)->conv0_capture_dqbuf_thread_callback(v4l2_buf, buffer, shared_buffer);
}

/*static*/ bool NvEncSession::encoder_capture_plane_dq_callback_thunk(struct v4l2_buffer *v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer, void *arg) {
  return reinterpret_cast<NvEncSession*>(arg)->encoder_capture_plane_dq_callback(v4l2_buf, buffer, shared_buffer);
}

