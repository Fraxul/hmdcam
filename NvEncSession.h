#pragma once
#include <functional>
#include <map>
#include <queue>
#include <pthread.h>
#include <stdint.h>

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
  void setFramerate(uint32_t numerator, uint32_t denominator) { m_framerateNumerator = numerator; m_framerateDenominator = denominator; }

  enum InputFormat {
    kInputFormatNV12,
    kInputFormatRGBX8,
  };

  void setInputFormat(InputFormat inputFormat) { m_inputFormat = inputFormat; }
  InputFormat inputFormat() const { return m_inputFormat; }

  size_t registerEncodedFrameDeliveryCallback(const std::function<void(const char*, size_t)>& cb);
  void unregisterEncodedFrameDeliveryCallback(size_t cbId);

  size_t inputFrameSize() const { return m_inputFrameSize; }
  bool submitFrame(char* data, size_t length, bool blockIfQueueFull);

  bool isRunning() const { return m_startCount > 0; }

  void start();
  void stop();

protected:

  uint32_t m_width, m_height;
  uint32_t m_bitsPerSecond;
  uint32_t m_framerateNumerator, m_framerateDenominator;
  uint32_t m_encoderPixfmt;
  InputFormat m_inputFormat;

  std::map<size_t, std::function<void(const char*, size_t)> > m_encodedFrameDeliveryCallbacks;
  size_t m_encodedFrameDeliveryCallbackIdGen;

  uint32_t m_startCount;
  bool m_inShutdown;

  pthread_mutex_t m_stateLock;
  pthread_mutex_t m_callbackLock;

  NvVideoEncoder* m_enc;
  NvVideoConverter* m_conv0;
  size_t m_inputFrameSize;

  std::queue<NvBuffer*> m_encoderOutputPlaneBufferQueue;
  pthread_mutex_t m_encoderOutputPlaneBufferQueueLock;
  pthread_cond_t m_encoderOutputPlaneBufferQueueCond;

  std::queue<NvBuffer*> m_conv0OutputPlaneBufferQueue;
  pthread_mutex_t m_conv0OutputPlaneBufferQueueLock;
  pthread_cond_t m_conv0OutputPlaneBufferQueueCond;

  static bool conv0_capture_dqbuf_thread_callback_thunk(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer, void* arg);
  static bool encoder_capture_plane_dq_callback_thunk(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer, void *arg);

  bool conv0_capture_dqbuf_thread_callback(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer);
  bool encoder_capture_plane_dq_callback(struct v4l2_buffer* v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer);
  bool encoder_output_plane_dq();
  bool conv0_output_plane_dq();

private:
  // noncopyable
  NvEncSession(const NvEncSession&);
  NvEncSession& operator=(const NvEncSession&);
};

