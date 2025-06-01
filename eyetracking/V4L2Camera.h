#pragma once
#include <vector>
#include <epoxy/egl.h>
#include <linux/videodev2.h>
#include <opencv2/core.hpp>
#include <boost/noncopyable.hpp>
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvmedia_ijpd.h"

class V4L2Camera : boost::noncopyable {
public:
  V4L2Camera();
  ~V4L2Camera();

  uint32_t streamWidth() const { return m_streamWidth; }
  uint32_t streamHeight() const { return m_streamHeight; }

  bool tryOpenSensor(const char* deviceFn);
  bool readFrame();

  uint64_t sensorTimestamp() const { return m_sensorTimestamp; }

  const cv::Mat& lumaPlane() const { return m_lumaPlane; }

  // ====================

  struct Buffer {
    struct v4l2_buffer vbuf = {0};
    void* mmap_ptr = nullptr;
    size_t mmap_length = 0;
    NvMediaBitstreamBuffer bitstream;

    // cv::Mat cvMat;
  };

  Buffer& currentBuffer() { return m_buffers[m_currentBufferIdx]; }
  const Buffer& currentBuffer() const { return m_buffers[m_currentBufferIdx]; }

protected:
  bool tryIoctl_(unsigned long request, const char* requestStr, void *param);
  void closeDevice();

  uint64_t m_tscToMonotonicRawOffset = 0;

  int m_fd = -1;
  struct v4l2_format m_fmt = {0};
  uint32_t m_streamWidth = 0;
  uint32_t m_streamHeight = 0;

  std::vector<Buffer> m_buffers;

  size_t m_currentBufferIdx = 0;

  uint64_t m_sensorTimestamp = 0;

  NvMediaIJPD* m_ijpd = nullptr;
  NvSciSyncModule m_syncModule = nullptr;
  NvSciBufModule m_bufModule = nullptr;

  uint32_t m_jpMaxWidth = 0;
  uint32_t m_jpMaxHeight = 0;
  uint32_t m_jpMaxBitstreamBytes = 0;

  NvSciBufAttrList populateOutputImageBufAttrList(uint32_t width, uint32_t height);
  NvSciBufAttrList m_outputImageAttrList = nullptr;

  NvSciBufObj m_outputBufObj = nullptr;
  NvSciSyncObj m_waiter = nullptr;
  NvSciSyncCpuWaitContext m_cpuWaitCtx = nullptr;
  NvSciSyncFence m_eofFence = NvSciSyncFenceInitializer;

  cv::Mat m_lumaPlane;

};

