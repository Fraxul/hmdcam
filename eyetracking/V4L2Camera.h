#pragma once
#include <string>
#include <vector>
#include <epoxy/egl.h>
#include <linux/videodev2.h>
// #include <opencv2/core.hpp>
#include <boost/noncopyable.hpp>

class V4L2Camera : boost::noncopyable {
public:
  V4L2Camera(const std::string& deviceFn);
  ~V4L2Camera();

  uint32_t streamWidth() const { return m_streamWidth; }
  uint32_t streamHeight() const { return m_streamHeight; }

  bool tryOpenSensor();
  bool readFrame();

  uint64_t sensorTimestamp() const { return m_sensorTimestamp; }

  // ====================

  struct Buffer {
    struct v4l2_buffer vbuf = {0};
    void* mmap_ptr = nullptr;
    size_t mmap_length = 0;

    // cv::Mat cvMat;
  };

  Buffer& currentBuffer() { return m_buffers[m_currentBufferIdx]; }
  const Buffer& currentBuffer() const { return m_buffers[m_currentBufferIdx]; }

protected:
  uint64_t m_tscToMonotonicRawOffset = 0;


  std::string m_deviceFn;
  int m_fd = -1;
  struct v4l2_format m_fmt = {0};
  uint32_t m_streamWidth = 0;
  uint32_t m_streamHeight = 0;

  std::vector<Buffer> m_buffers;

  int m_returnBufferIdx = -1;
  size_t m_currentBufferIdx = 0;

  uint64_t m_sensorTimestamp = 0;

  void queueBufferAtIndex(size_t bufferIdx);
};

