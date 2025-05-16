#include "V4L2Camera.h"
#include "common/Timing.h"
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <string.h>
#include "rhi/cuda/CudaUtil.h"
#include "rhi/RHI.h"

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)
#define CHECK_ZERO(x) if ((x) != 0) { fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); abort(); }
#define CHECK_TRUE(x) if (!(x)) { fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); abort(); }
#define CHECK_NOT_NULL(x) if ((x) == NULL) { fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); abort(); }

static const size_t kBufferCount = 4;

V4L2Camera::V4L2Camera(const std::string& deviceFn) : m_deviceFn(deviceFn) {
#if 0
  // Read offset between TSC (used for V4L2 timestamps) and CLOCK_MONOTONIC_RAW.
  // This shouldn't change unless the system gets suspended and resumed.
  // We don't support keeping a capture session open across suspend/resume.
  {
    const char* offset_ns_path = "/sys/devices/system/clocksource/clocksource0/offset_ns";
    int fd = open(offset_ns_path, O_RDONLY);
    if (fd < 0)
      die("Couldn't open %s: %s", offset_ns_path, strerror(errno));
    char buf[64];
    ssize_t res = read(fd, buf, 64);
    if (res < 0)
      die("Couldn't read from %s: %s", offset_ns_path, strerror(errno));

    buf[res] = '\0';
    m_tscToMonotonicRawOffset = strtoul(buf, NULL, 10);
    close(fd);
    printf("TSC offset: %lu\n", m_tscToMonotonicRawOffset);
  }
#else
  m_tscToMonotonicRawOffset = 0;
#endif

}


bool V4L2Camera::tryOpenSensor() {
  {
    if (m_fd >= 0)
      return true; // Sensor is already open

    m_fd = ::open(m_deviceFn.c_str(), O_RDWR);
    if (m_fd < 0) {
      fprintf(stderr, "open(%s): %s\n", m_deviceFn.c_str(), strerror(errno));
      return false;
    }

    struct v4l2_capability vcap;
    int err = ioctl(m_fd, VIDIOC_QUERYCAP, &vcap);
    if (err < 0) {
      fprintf(stderr, "ioctl(%s, VIDIOC_QUERYCAP): %s\n", m_deviceFn.c_str(), strerror(errno));
      goto err;
    }

    printf("device %s:\n", m_deviceFn.c_str());
    printf("  driver: %s\n", reinterpret_cast<const char*>(vcap.driver));
    printf("  card: %s\n", reinterpret_cast<const char*>(vcap.card));
    printf("  bus_info: %s\n", reinterpret_cast<const char*>(vcap.bus_info));

    // Get pixel format
    m_fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(m_fd, VIDIOC_G_FMT, &m_fmt) < 0) {
      fprintf(stderr, "Failed to get camera output format: %s (%d)\n", strerror(errno), errno);
      goto err;
    }

    printf("Camera ouput format (before modeset): (%d x %d) pixfmt: %.4s stride: %d, imagesize: %d\n",
            m_fmt.fmt.pix.width,
            m_fmt.fmt.pix.height,
            (const char*) (&m_fmt.fmt.pix.pixelformat),
            m_fmt.fmt.pix.bytesperline,
            m_fmt.fmt.pix.sizeimage);

    // Adjust format
    m_fmt.fmt.pix.width = 1280;
    m_fmt.fmt.pix.height = 800;
    m_fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    m_fmt.fmt.pix.bytesperline = 0; // Meaningless for MJPEG
    m_fmt.fmt.pix.sizeimage = 0; // Get the value from the driver

    if (ioctl(m_fd, VIDIOC_S_FMT, &m_fmt) < 0) {
      fprintf(stderr, "Failed to issue VIDIOC_S_FMT: %s", strerror(errno));
      goto err;
    }

    if (ioctl(m_fd, VIDIOC_G_FMT, &m_fmt) < 0) {
      fprintf(stderr, "Failed to get camera output format: %s (%d)\n", strerror(errno), errno);
      goto err;
    }

    printf("Camera ouput format (after modeset): (%d x %d) pixfmt: %.4s stride: %d, imagesize: %d\n",
            m_fmt.fmt.pix.width,
            m_fmt.fmt.pix.height,
            (const char*) (&m_fmt.fmt.pix.pixelformat),
            m_fmt.fmt.pix.bytesperline,
            m_fmt.fmt.pix.sizeimage);

    m_streamWidth = m_fmt.fmt.pix.width;
    m_streamHeight = m_fmt.fmt.pix.height;

    // Adjust frame interval

    struct v4l2_streamparm sp;
    memset(&sp, 0, sizeof(sp));
    sp.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;


    if (ioctl(m_fd, VIDIOC_G_PARM, &sp) < 0) {
      fprintf(stderr, "Failed to get stream params: %s (%d)\n", strerror(errno), errno);
      goto err;
    }

    printf("Capture params: caps 0x%x, interval %u/%u, extmode = 0x%x, readbuffers = %u\n",
      sp.parm.capture.capability,
      sp.parm.capture.timeperframe.numerator, sp.parm.capture.timeperframe.denominator,
      sp.parm.capture.extendedmode, sp.parm.capture.readbuffers);


    // Adjust framerate. TODO: this should be configurable
    sp.parm.capture.timeperframe.numerator = 1;
    sp.parm.capture.timeperframe.denominator = 30;

    if (ioctl(m_fd, VIDIOC_S_PARM, &sp) < 0) {
      fprintf(stderr, "Failed to set stream params: %s (%d)\n", strerror(errno), errno);
      goto err;
    }

    // Stream params will be updated by the VIDIOC_S_PARM ioctl
    printf("Capture params: caps 0x%x, interval %u/%u, extmode = 0x%x, readbuffers = %u\n",
      sp.parm.capture.capability,
      sp.parm.capture.timeperframe.numerator, sp.parm.capture.timeperframe.denominator,
      sp.parm.capture.extendedmode, sp.parm.capture.readbuffers);


    // Request MMAP buffers
    struct v4l2_requestbuffers reqbuf;

    memset(&reqbuf, 0, sizeof (reqbuf));
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbuf.memory = V4L2_MEMORY_MMAP;
    reqbuf.count = kBufferCount;

    if (ioctl(m_fd, VIDIOC_REQBUFS, &reqbuf) < 0) {
      fprintf(stderr, "VIDIOC_REQBUFS failed: %s\n", strerror(errno));
      goto err;
    }

    assert(reqbuf.count == kBufferCount);

    // Set up buffer allocations

    m_buffers.resize(kBufferCount);
    for (size_t bufIdx = 0; bufIdx < kBufferCount; ++bufIdx) {
      Buffer& b = m_buffers[bufIdx];
      // b.bayerCudaBufferSize = m_fmt.fmt.pix.bytesperline * m_fmt.fmt.pix.height;
      // b.bayerCudaBufferStrideBytes = m_fmt.fmt.pix.bytesperline;

      // b.cvMat.create(m_streamHeight, m_streamWidth, CV_8U);

      // Query v4l2 buffers
      memset(&b.vbuf, 0, sizeof(b.vbuf));
      b.vbuf.index = bufIdx;
      b.vbuf.type = reqbuf.type;
      b.vbuf.memory = reqbuf.memory;
      if (ioctl(m_fd, VIDIOC_QUERYBUF, &b.vbuf) < 0) {
        fprintf(stderr, "VIDIOC_QUERYBUF failed: %s\n", strerror(errno));
        goto err;
      }
      printf("buf[%zu].length = %u\n", bufIdx, b.vbuf.length);

      b.mmap_length = b.vbuf.length;
      b.mmap_ptr = mmap(nullptr, b.mmap_length, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, b.vbuf.m.offset);
      assert(b.mmap_ptr != MAP_FAILED);
    }

    // Queue all buffers
    for (size_t bufferIdx = 0; bufferIdx < m_buffers.size(); ++bufferIdx) {
      queueBufferAtIndex(bufferIdx);
    }

    // Reset return-buffer index, since we start without any buffers to return
    m_returnBufferIdx = -1;

    // STREAMON
    int streamType = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(m_fd, VIDIOC_STREAMON, &streamType) < 0) {
      fprintf(stderr, "VIDIOC_STREAMON failed: %s\n", strerror(errno));
      goto err;
    }

    return true;
  }

err:
  close(m_fd);
  m_fd = -1;
  return false;
}

void V4L2Camera::queueBufferAtIndex(size_t bufferIdx) {
  auto& buffer = m_buffers[bufferIdx];
  //buffer.vbuf.index = bufferIdx;
  //buffer.vbuf.m.offset = 0;
  //buffer.vbuf.m.userptr = (unsigned long) buffer.bayerCudaPtr;
  //buffer.vbuf.length = buffer.bayerCudaBufferSize;
  if (ioctl(m_fd, VIDIOC_QBUF, &buffer.vbuf) < 0) {
    die("VIDIOC_QBUF failed: %s", strerror(errno));
  }
}

bool V4L2Camera::readFrame() {
  // If there's a buffer to return, do it now
  if (m_returnBufferIdx >= 0) {
    queueBufferAtIndex(m_returnBufferIdx);
    m_returnBufferIdx = -1;
  }

  struct v4l2_buffer dqbuf = {0};
  dqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  dqbuf.memory = V4L2_MEMORY_MMAP;
  if (ioctl(m_fd, VIDIOC_DQBUF, &dqbuf) < 0) {
    fprintf(stderr, "readFrame: %s\n", strerror(errno));
    return false;
  }

  printf("dqbuf index=%d bytesused=%d\n", dqbuf.index, dqbuf.bytesused);

  m_currentBufferIdx = dqbuf.index;
  m_returnBufferIdx = m_currentBufferIdx; // don't forget to return the buffer!

  // Fill out metadata
  m_sensorTimestamp = static_cast<uint64_t>((dqbuf.timestamp.tv_sec * 1'000'000'000ULL) + (dqbuf.timestamp.tv_usec * 1'000ULL)) - m_tscToMonotonicRawOffset;

  // TODO mjpeg decompress
  Buffer& b = currentBuffer();

  return true;
}

V4L2Camera::~V4L2Camera() {
  int streamType = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(m_fd, VIDIOC_STREAMOFF, &streamType) < 0) {
    fprintf(stderr, "VIDIOC_STREAMOFF failed: %s", strerror(errno));
  }

  if (m_fd >= 0)
    ::close(m_fd);
}

