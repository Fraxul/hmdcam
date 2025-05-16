#include "V4L2Camera.h"
#include "common/tegra/NvSciUtil.h"
#include "common/Timing.h"
#include "nvmedia_ijpd.h"
#include "nvmedia_common_encode_decode.h"
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
#define CHECK_PTR(x) if (!(x)) { fprintf(stderr, "%s:%d: %s failed (returned NULL)\n", __FILE__, __LINE__, #x); abort(); }
#define CleanupPtr(Fn, Obj, ... ) if (Obj != nullptr) { Fn(Obj  ,## __VA_ARGS__); Obj = nullptr; }

static const size_t kBufferCount = 4;


NvSciBufAttrList V4L2Camera::populateOutputImageBufAttrList(uint32_t width, uint32_t height) {
  NvSciBufAttrList reconciledOutputImageAttrList = nullptr;
  {

    NvSciBufAttrList outputImageAttrList = nullptr;
    NVSCI_CHECK(NvSciBufAttrListCreate(m_bufModule, &outputImageAttrList));
    NVMEDIA_CHECK(NvMediaIJPDFillNvSciBufAttrList(NVMEDIA_JPEG_INSTANCE_0, outputImageAttrList));

    NvSciBufType bufType = NvSciBufType_Image;
    NvSciBufAttrValImageLayoutType layout = NvSciBufImage_PitchLinearType;
    NvSciBufAttrValAccessPerm perm = NvSciBufAccessPerm_Readonly;
    NvSciBufAttrValImageScanType scantype = NvSciBufScan_ProgressiveType;

    NvSciBufSurfType surftype = NvSciSurfType_YUV;
    NvSciBufSurfMemLayout memlayout = NvSciSurfMemLayout_SemiPlanar;
    NvSciBufSurfSampleType sampletype = NvSciSurfSampleType_420;
    NvSciBufSurfBPC bpc = NvSciSurfBPC_8;
    NvSciBufSurfComponentOrder componentorder = NvSciSurfComponentOrder_YUV;

    bool cpuAccessFlag = true;

    NvSciBufAttrKeyValuePair imgBufAttrs[] = {
      {NvSciBufGeneralAttrKey_Types, &bufType, sizeof(bufType)},
      {NvSciBufImageAttrKey_Layout, &layout, sizeof(layout)},
      {NvSciBufImageAttrKey_ScanType, &scantype, sizeof(scantype)},
      {NvSciBufGeneralAttrKey_NeedCpuAccess, &cpuAccessFlag, sizeof(cpuAccessFlag)},
      {NvSciBufGeneralAttrKey_RequiredPerm, &perm, sizeof(perm)},

      {NvSciBufImageAttrKey_SurfType, &surftype, sizeof(surftype)},
      {NvSciBufImageAttrKey_SurfMemLayout, &memlayout, sizeof(memlayout)},
      {NvSciBufImageAttrKey_SurfSampleType, &sampletype, sizeof(sampletype)},
      {NvSciBufImageAttrKey_SurfBPC, &bpc, sizeof(bpc)},
      {NvSciBufImageAttrKey_SurfComponentOrder, &componentorder, sizeof(componentorder)},
      {NvSciBufImageAttrKey_SurfWidthBase, &width, sizeof(width)},
      {NvSciBufImageAttrKey_SurfHeightBase, &height, sizeof(height)},
    };

    NVSCI_CHECK(NvSciBufAttrListSetAttrs(outputImageAttrList, imgBufAttrs, sizeof(imgBufAttrs) / sizeof(NvSciBufAttrKeyValuePair)));

    NvSciBufAttrList bufferConflictList = nullptr;
    NVSCI_CHECK(NvSciBufAttrListReconcile(&outputImageAttrList, 1, &reconciledOutputImageAttrList, &bufferConflictList));

    assert(reconciledOutputImageAttrList != nullptr);
    bool isReconciled = false;
    NVSCI_CHECK(NvSciBufAttrListIsReconciled(reconciledOutputImageAttrList, &isReconciled));
    assert(isReconciled);

    NvSciBufAttrListFree(outputImageAttrList);
  }
  return reconciledOutputImageAttrList;
}


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

  NvMediaVersion ijpdVersion;
  memset(&ijpdVersion, 0, sizeof(ijpdVersion));
  NVMEDIA_CHECK(NvMediaIJPDGetVersion(&ijpdVersion));
  printf("IJPD version: %u.%u.%u\n", ijpdVersion.major, ijpdVersion.minor, ijpdVersion.patch);
  if ( (ijpdVersion.major != NVMEDIA_IJPD_VERSION_MAJOR)
    || (ijpdVersion.minor != NVMEDIA_IJPD_VERSION_MINOR)
    || (ijpdVersion.patch != NVMEDIA_IJPD_VERSION_PATCH)) {

    printf("WARNING: NvMediaIJPD header version mismatch -- expected %u.%u.%u\n",
      NVMEDIA_IJPD_VERSION_MAJOR,
      NVMEDIA_IJPD_VERSION_MINOR,
      NVMEDIA_IJPD_VERSION_PATCH);
  }

  NVSCI_CHECK(NvSciSyncModuleOpen(&m_syncModule));
  NVSCI_CHECK(NvSciBufModuleOpen(&m_bufModule));

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

    // Create or resize NvMediaIJPD JPEG decoder
    if (!m_ijpd) {
      CHECK_PTR(m_ijpd = NvMediaIJPDCreate(m_streamWidth, m_streamHeight, m_fmt.fmt.pix.sizeimage, /*supportPartialAccel=*/ false, NVMEDIA_JPEG_INSTANCE_0));


      // Create sync objs on first ijpd alloc

      // CPU Signaler / IJPD Waiter
      {
        NvSciSyncAttrList attrList = nullptr;
        NVSCI_CHECK(NvSciSyncAttrListCreate(m_syncModule, &attrList));
        NVMEDIA_CHECK(NvMediaIJPDFillNvSciSyncAttrList(m_ijpd, attrList, NVMEDIA_WAITER));

        NvSciSyncAttrList reconciledList = ReconcileNvSciSyncAttrLists(attrList, CreateNvSciSyncCpuSignalerAttrList(m_syncModule)); // frees attrList

        NVSCI_CHECK(NvSciSyncObjAlloc(reconciledList, &m_signaler));
        NVMEDIA_CHECK(NvMediaIJPDRegisterNvSciSyncObj(m_ijpd, NVMEDIA_PRESYNCOBJ, m_signaler));

        NvSciSyncAttrListFree(reconciledList);
      }

      // CPU Waiter (with CPU waiter context) / IJPD Signaler
      {
        NVSCI_CHECK(NvSciSyncCpuWaitContextAlloc(m_syncModule, &m_cpuWaitCtx));

        NvSciSyncAttrList attrList;
        NVSCI_CHECK(NvSciSyncAttrListCreate(m_syncModule, &attrList));
        NVMEDIA_CHECK(NvMediaIJPDFillNvSciSyncAttrList(m_ijpd, attrList, NVMEDIA_SIGNALER));

        NvSciSyncAttrList reconciledList = ReconcileNvSciSyncAttrLists(attrList, CreateNvSciSyncCpuWaiterAttrList(m_syncModule)); // frees attrList

        NVSCI_CHECK(NvSciSyncObjAlloc(reconciledList, &m_waiter));
        NVMEDIA_CHECK(NvMediaIJPDRegisterNvSciSyncObj(m_ijpd, NVMEDIA_EOFSYNCOBJ, m_waiter));
        NVMEDIA_CHECK(NvMediaIJPDSetNvSciSyncObjforEOF(m_ijpd, m_waiter));
        NvSciSyncAttrListFree(reconciledList);
      }

    } else {
      if ((m_streamWidth > m_jpMaxWidth) || (m_streamHeight > m_jpMaxHeight) || (m_fmt.fmt.pix.bytesperline > m_jpMaxBitstreamBytes)) {
        // Resize, which requires reallocating the output buffer
        NVMEDIA_CHECK(NvMediaIJPDResize(m_ijpd, m_streamWidth, m_streamHeight, m_fmt.fmt.pix.sizeimage));

        CleanupPtr(NvSciBufAttrListFree, m_outputImageAttrList);
        CleanupPtr(NvSciBufObjFree, m_outputBufObj);
      }
    }
    m_jpMaxWidth = m_streamWidth;
    m_jpMaxHeight = m_streamHeight;
    m_jpMaxBitstreamBytes = m_fmt.fmt.pix.sizeimage;
    if (!m_outputImageAttrList) {
      m_outputImageAttrList = populateOutputImageBufAttrList(m_streamWidth, m_streamHeight);
    }

    if (!m_outputBufObj) {
      NVSCI_CHECK(NvSciBufObjAlloc(m_outputImageAttrList, &m_outputBufObj));
      NVMEDIA_CHECK(NvMediaIJPDRegisterNvSciBufObj(m_ijpd, m_outputBufObj));
      const uint8_t* outputBufPtr = nullptr;
      NVSCI_CHECK(NvSciBufObjGetConstCpuPtr(m_outputBufObj, reinterpret_cast<const void**>(&outputBufPtr)));

      NvSciBufAttrKeyValuePair attrs[] = {
        {NvSciBufImageAttrKey_PlaneCount, nullptr, 0},
        {NvSciBufImageAttrKey_PlaneOffset, nullptr, 0},
        {NvSciBufImageAttrKey_PlaneWidth, nullptr, 0},
        {NvSciBufImageAttrKey_PlaneHeight, nullptr, 0},
        {NvSciBufImageAttrKey_PlaneDatatype, nullptr, 0},
        {NvSciBufImageAttrKey_PlaneChannelCount, nullptr, 0},
        {NvSciBufImageAttrKey_PlanePitch, nullptr, 0},
      };
      NVSCI_CHECK(NvSciBufAttrListGetAttrs(m_outputImageAttrList, attrs, sizeof(attrs)/sizeof(attrs[0])));

#if 0
      for (uint32_t plane = 0; plane < reinterpret_cast<const uint32_t*>(attrs[0].value)[0]; ++plane) {

        printf("Offset=%lu w=%u h=%u datatype=%u channels=%u pitch=%u\n",
          reinterpret_cast<const uint64_t*>(attrs[1].value)[plane],
          reinterpret_cast<const uint32_t*>(attrs[2].value)[plane],
          reinterpret_cast<const uint32_t*>(attrs[3].value)[plane],
          reinterpret_cast<const uint32_t*>(attrs[4].value)[plane],
          reinterpret_cast<const uint8_t*>(attrs[5].value)[plane],
          reinterpret_cast<const uint32_t*>(attrs[6].value)[plane]);
      }
#endif

      // Luma plane wrapper
      {
        constexpr size_t plane = 0;
        auto planeOffset = reinterpret_cast<const uint64_t*>(attrs[1].value)[plane];
        auto planeWidth = reinterpret_cast<const uint32_t*>(attrs[2].value)[plane];
        auto planeHeight = reinterpret_cast<const uint32_t*>(attrs[3].value)[plane];

        auto planeDatatype = reinterpret_cast<const NvSciBufAttrValDataType*>(attrs[4].value)[plane];
        assert(planeDatatype == NvSciDataType_Uint8);

        auto planeChannels = reinterpret_cast<const uint8_t*>(attrs[5].value)[plane];
        assert(planeChannels == 1);

        auto planePitch = reinterpret_cast<const uint32_t*>(attrs[6].value)[plane];
        m_lumaPlane = cv::Mat(planeHeight, planeWidth, CV_8UC1, const_cast<uint8_t*>(outputBufPtr + planeOffset), /*pitchBytes=*/ planePitch);
      }
    }


    // TODO: Factor out CPU-accessible sync and buffer management from cudla-standalone

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

      b.bitstream.bitstream = (uint8_t*) b.mmap_ptr;
      b.bitstream.bitstreamSize = b.mmap_length;
      b.bitstream.bitstreamBytes = 0; // Will be set when buffer is dequeued

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

  // printf("dqbuf index=%d bytesused=%d\n", dqbuf.index, dqbuf.bytesused);

  m_currentBufferIdx = dqbuf.index;
  m_returnBufferIdx = m_currentBufferIdx; // don't forget to return the buffer!

  // Fill out metadata
  m_sensorTimestamp = static_cast<uint64_t>((dqbuf.timestamp.tv_sec * 1'000'000'000ULL) + (dqbuf.timestamp.tv_usec * 1'000ULL)) - m_tscToMonotonicRawOffset;

  // Submit MJPEG decompress
  Buffer& b = currentBuffer();
  b.bitstream.bitstreamBytes = dqbuf.bytesused;


  // uint64_t startTime = currentTimeNs();

  NVMEDIA_CHECK(NvMediaIJPDInsertPreNvSciSyncFence(m_ijpd, &m_preFence));
  NVMEDIA_CHECK(NvMediaIJPDRenderYUV(m_ijpd, m_outputBufObj,
    /*downscaleLog2=*/ 0,
    /*numBitstreamBuffers=*/ 1, &b.bitstream,
    /*flags=*/ 0,
    NVMEDIA_JPEG_INSTANCE_0));

  // Signal wait events
  NvSciSyncObjSignal(m_signaler);

  // Wait for operations to finish and bring output buffer to CPU.
  NVMEDIA_CHECK(NvMediaIJPDGetEOFNvSciSyncFence(m_ijpd, m_waiter, &m_eofFence));
  NVSCI_CHECK(NvSciSyncFenceWait(&m_eofFence, m_cpuWaitCtx, -1));

  // printf("MJPEG decompress took %.3fms\n", deltaTimeMs(startTime, currentTimeNs()));

#if 0
  NVMEDIAJPEGDecInfo decInfo;
  NvMediaStatus st = NvMediaIJPDGetInfo(&decInfo, 1, &b.bitstream);
  if (st == NVMEDIA_STATUS_OK) {
    printf("info w=%u h=%u partialAccel=%u numAppMarkers=%u\n", decInfo.width, decInfo.height, decInfo.partialAccel, decInfo.num_app_markers);
  } else {
    printf("bad nvmedia status 0x%x\n", st);
  }
#endif

  return true;
}

V4L2Camera::~V4L2Camera() {
  int streamType = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (ioctl(m_fd, VIDIOC_STREAMOFF, &streamType) < 0) {
    fprintf(stderr, "VIDIOC_STREAMOFF failed: %s", strerror(errno));
  }

  if (m_fd >= 0)
    ::close(m_fd);

  if (m_signaler)
    NvMediaIJPDUnregisterNvSciSyncObj(m_ijpd, m_signaler);
  if (m_waiter)
    NvMediaIJPDUnregisterNvSciSyncObj(m_ijpd, m_waiter);
  if (m_outputBufObj)
    NvMediaIJPDUnregisterNvSciBufObj(m_ijpd, m_outputBufObj);

  CleanupPtr(NvSciBufAttrListFree, m_outputImageAttrList);
  CleanupPtr(NvSciSyncObjFree, m_signaler);
  CleanupPtr(NvSciSyncObjFree, m_waiter);
  CleanupPtr(NvSciSyncCpuWaitContextFree, m_cpuWaitCtx);


  CleanupPtr(NvMediaIJPDDestroy, m_ijpd);
  CleanupPtr(NvSciSyncModuleClose, m_syncModule);
  CleanupPtr(NvSciBufModuleClose, m_bufModule);
}

