#include "DebugCameraProvider.h"
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/Timing.h"
#include "common/VPIUtil.h"
#include "common/glmCvInterop.h"
#include "common/remapArray.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/cuda/RHICVInterop.h"
#include "rhi/gl/GLCommon.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <string>
#include <assert.h>
#include <cuda.h>
#include <npp.h>
#include <epoxy/gl.h> // epoxy_is_desktop_gl

#include <arpa/inet.h>
#include <errno.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#define PER_EYE for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)

const int kPort = 55443;

bool safe_read(int fd, void* buffer, size_t length) {
  uint8_t* p = reinterpret_cast<uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining) {
    ssize_t res = read(fd, p, remaining);
    if (res < 0) {
      return false;
    }

    p += res;
    remaining -= res;
  }
  return true;
}

DebugCameraProvider::DebugCameraProvider() : DepthMapGenerator(kDepthBackendMock) {

}

bool DebugCameraProvider::connect(const char* debugHost) {
  if (m_fd >= 0) {
    close(m_fd);
    m_fd = -1;
  }

  // Resolve host / establish connection
  {
    struct addrinfo hints;
    struct addrinfo *result = nullptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC; // Allow IPv4 or IPv6
    hints.ai_socktype = SOCK_STREAM;

    char portBuf[16];
    sprintf(portBuf, "%u", kPort);

    int s = getaddrinfo(debugHost, portBuf, &hints, &result);
    if (s != 0) {
      fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(s));
      return false;
    }

    // Walk the result list and try to connect to each in turn
    for (struct addrinfo* rp = result; rp != NULL; rp = rp->ai_next) {
      m_fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
      if (m_fd == -1)
        continue;

      if (::connect(m_fd, rp->ai_addr, rp->ai_addrlen) != -1)
        break; // Connection succeeded

      // Failure case
      close(m_fd);
      m_fd = -1;
    }

    freeaddrinfo(result);
    if (m_fd < 0) {
      fprintf(stderr, "Connection failed\n");
      return false;
    }
  }

  // Connection OK, read data header
  SerializationBuffer cfg;
  cv::FileStorage csConfigFs;
  {
    uint32_t streamHeaderLen = 0;
    uint32_t tmp = 0;
    if (!safe_read(m_fd, &tmp, sizeof(tmp)))
      return false;
    streamHeaderLen = boost::endian::big_to_native<uint32_t>(tmp);

    printf("Stream header length: %u bytes\n", streamHeaderLen);
    std::string streamHeader;
    streamHeader.resize(streamHeaderLen);
    if (!safe_read(m_fd, const_cast<char*>(streamHeader.data()), streamHeader.size()))
      return false;
    cfg = SerializationBuffer(streamHeader);

    if (!safe_read(m_fd, &tmp, sizeof(tmp)))
      return false;

    uint32_t csConfigLen = boost::endian::big_to_native<uint32_t>(tmp);
    printf("CameraSystem configuration length: %u bytes\n", csConfigLen);
    std::string csConfig;
    csConfig.resize(csConfigLen);
    if (!safe_read(m_fd, const_cast<char*>(csConfig.data()), csConfig.size()))
      return false;

    m_cameraSystemConfig = cv::String(csConfig.data(), csConfig.size());
  }

  cfg.rewind();
  m_streamCount = cfg.get_u32();
  m_streamWidth = cfg.get_u32();
  m_streamHeight = cfg.get_u32();

  printf("Stream header: %zu cameras, %ux%u\n", m_streamCount, m_streamWidth, m_streamHeight);
  m_eglColorFormat = (CUeglColorFormat) cfg.get_u32();

  memset(&m_lumaResourceDescriptor, 0, sizeof(m_lumaResourceDescriptor));
  m_lumaResourceDescriptor.resType = CU_RESOURCE_TYPE_PITCH2D;
  m_lumaResourceDescriptor.res.pitch2D.format = (CUarray_format) cfg.get_u32();
  m_lumaResourceDescriptor.res.pitch2D.numChannels = cfg.get_u32();
  m_lumaResourceDescriptor.res.pitch2D.width = cfg.get_u32();
  m_lumaResourceDescriptor.res.pitch2D.height = cfg.get_u32();
  m_lumaResourceDescriptor.res.pitch2D.pitchInBytes = cfg.get_u32();

  memset(&m_chromaResourceDescriptor, 0, sizeof(m_chromaResourceDescriptor));
  m_chromaResourceDescriptor.resType = CU_RESOURCE_TYPE_PITCH2D;
  m_chromaResourceDescriptor.res.pitch2D.format = (CUarray_format) cfg.get_u32();
  m_chromaResourceDescriptor.res.pitch2D.numChannels = cfg.get_u32();
  m_chromaResourceDescriptor.res.pitch2D.width = cfg.get_u32();
  m_chromaResourceDescriptor.res.pitch2D.height = cfg.get_u32();
  m_chromaResourceDescriptor.res.pitch2D.pitchInBytes = cfg.get_u32();

  printf("  Luma: %.4zu x %.4zu NumChannels=%u Format=%x pitchInBytes=%zu\n",
    m_lumaResourceDescriptor.res.pitch2D.width, m_lumaResourceDescriptor.res.pitch2D.height,
    m_lumaResourceDescriptor.res.pitch2D.numChannels,
    m_lumaResourceDescriptor.res.pitch2D.format,
    m_lumaResourceDescriptor.res.pitch2D.pitchInBytes);
  printf("Chroma: %.4zu x %.4zu NumChannels=%u Format=%x pitchInBytes=%zu\n",
    m_chromaResourceDescriptor.res.pitch2D.width, m_chromaResourceDescriptor.res.pitch2D.height,
    m_chromaResourceDescriptor.res.pitch2D.numChannels,
    m_chromaResourceDescriptor.res.pitch2D.format,
    m_chromaResourceDescriptor.res.pitch2D.pitchInBytes);

  m_lumaPlaneSizeBytes = m_lumaResourceDescriptor.res.pitch2D.height * m_lumaResourceDescriptor.res.pitch2D.pitchInBytes;
  m_chromaPlaneSizeBytes = m_chromaResourceDescriptor.res.pitch2D.height * m_chromaResourceDescriptor.res.pitch2D.pitchInBytes;

  // Depth map generator settings
  m_stereoViewCount = cfg.get_u32();
  if (m_stereoViewCount) {
    m_disparityWidth = cfg.get_u32();
    m_disparityHeight = cfg.get_u32();

    m_algoDownsampleX = cfg.get_u32();
    m_algoDownsampleY = cfg.get_u32();

    m_maxDisparity = cfg.get_u32();
    m_disparityPrescale = cfg.get_float();
    m_useFP16Disparity = cfg.get_u32();

    m_stereoDisparitySizeBytes = m_disparityWidth * m_disparityHeight * sizeof(uint16_t);

    printf("Disp: %ux%u Downsample: %ux%u maxDisp=%u prescale=%f useFP16=%u sizeBytes=%u\n",
      m_disparityWidth, m_disparityHeight, m_algoDownsampleX, m_algoDownsampleY,
      m_maxDisparity, m_disparityPrescale, m_useFP16Disparity, m_stereoDisparitySizeBytes);

    m_stereoDisparityRecvMats.resize(m_stereoViewCount);
    for (uint32_t viewIdx = 0; viewIdx < m_stereoViewCount; ++viewIdx) {
      m_stereoDisparityRecvMats[viewIdx].create(m_disparityHeight, m_disparityWidth, CV_16UC1);
    }
  }


  assert(m_streamCount && m_streamWidth && m_streamHeight);

  // Allocate format conversion temporary
  m_gpuMatRGBTmp.create(m_streamHeight, m_streamWidth, CV_8UC3);

  // Setup host buffers and RHISurfaces
  m_streamData.resize(streamCount());
  for (size_t streamIdx = 0; streamIdx < streamCount(); ++streamIdx) {
    StreamData& sd = m_streamData[streamIdx];

    CUDA_CHECK(cuMemHostAlloc(&sd.hostLumaBuffer, m_lumaResourceDescriptor.res.pitch2D.pitchInBytes * m_lumaResourceDescriptor.res.pitch2D.height, /*flags=*/ 0));
    CUDA_CHECK(cuMemHostAlloc(&sd.hostChromaBuffer, m_chromaResourceDescriptor.res.pitch2D.pitchInBytes * m_chromaResourceDescriptor.res.pitch2D.height, /*flags=*/ 0));

    sd.rhiSurfaceRGBA = rhi()->newTexture2D(streamWidth(), streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
  }

  // Ensure gpumats are created
  updateSurfaces();

  // Wrap GpuMats to create CUtexObjects
  for (size_t streamIdx = 0; streamIdx < streamCount(); ++streamIdx) {
    StreamData& sd = m_streamData[streamIdx];
    // Luma texObject

    {
      m_lumaResourceDescriptor.res.pitch2D.devPtr = (CUdeviceptr) sd.gpuMatLuma.cudaPtr();

      CUDA_TEXTURE_DESC texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
      texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
      texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
      texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
      // texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES; // optional
      texDesc.maxAnisotropy = 1;

      CUDA_CHECK(cuTexObjectCreate(&sd.cudaLumaTexObject, &m_lumaResourceDescriptor, &texDesc, /*resourceViewDescriptor=*/ nullptr));
    }

    // Chroma texObject
    {
      m_chromaResourceDescriptor.res.pitch2D.devPtr = (CUdeviceptr) sd.gpuMatChroma.cudaPtr();

      CUDA_TEXTURE_DESC texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
      texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
      texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
      texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
      // texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES; // optional
      texDesc.maxAnisotropy = 1;

      CUDA_CHECK(cuTexObjectCreate(&sd.cudaChromaTexObject, &m_chromaResourceDescriptor, &texDesc, /*resourceViewDescriptor=*/ nullptr));
    }
  }

#ifdef HAVE_VPI2
  // Wrap GpuMats to create VPIImages
  for (size_t streamIdx = 0; streamIdx < m_streamCount; ++streamIdx) {
    StreamData& sd = m_streamData[streamIdx];

    VPIImageData data;
    memset(&data, 0, sizeof(VPIImageData));
    data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    data.buffer.pitch.format = VPI_IMAGE_FORMAT_Y8_ER;
    data.buffer.pitch.numPlanes = 1;
    data.buffer.pitch.planes[0].pixelType = vpiImageFormatGetPlanePixelType(data.buffer.pitch.format, 0);
    data.buffer.pitch.planes[0].width = m_streamWidth;
    data.buffer.pitch.planes[0].height = m_streamHeight;
    data.buffer.pitch.planes[0].pitchBytes = sd.gpuMatLuma.step;
    data.buffer.pitch.planes[0].data = sd.gpuMatLuma.cudaPtr();

    VPI_CHECK(vpiImageCreateWrapper(&data, /*params=*/ NULL, VPI_BACKEND_CUDA | VPI_REQUIRE_BACKENDS, &sd.vpiImage));
  }
#endif

  // Start the frame reader thread
  pthread_create(&m_streamThread, NULL, &streamThreadEntryPoint, (void*) this);
  return true;
}


void DebugCameraProvider::streamThreadFn() {

  pthread_mutex_lock(&m_frameConsumedMutex);
  while (true) {

    // Wait until the main thread consumes the previous frame
    while (m_streamFrameReadyToConsume) {
      pthread_cond_wait(&m_frameConsumedCond, &m_frameConsumedMutex);
    }
    pthread_mutex_unlock(&m_frameConsumedMutex);

    for (uint32_t streamIdx = 0; streamIdx < m_streamCount; ++streamIdx) {
      if (!safe_read(m_fd, m_streamData[streamIdx].hostLumaBuffer, m_lumaPlaneSizeBytes)) goto cleanup;
      if (!safe_read(m_fd, m_streamData[streamIdx].hostChromaBuffer, m_chromaPlaneSizeBytes)) goto cleanup;
    }
    for (uint32_t stereoViewIdx = 0; stereoViewIdx < m_stereoViewCount; ++stereoViewIdx) {
      if (!safe_read(m_fd, m_stereoDisparityRecvMats[stereoViewIdx].data, m_stereoDisparitySizeBytes)) goto cleanup;
    }

    pthread_mutex_lock(&m_frameConsumedMutex);
    m_streamFrameReadyToConsume = true;
  }

cleanup:
  fprintf(stderr, "DebugCameraProvider: stream thread exited\n");
}

DebugCameraProvider::~DebugCameraProvider() {
  if (m_fd >= 0)
    close(m_fd);

  for (StreamData& sd : m_streamData) {
    cuTexObjectDestroy(sd.cudaLumaTexObject);
    cuTexObjectDestroy(sd.cudaChromaTexObject);
#ifdef HAVE_VPI2
    vpiImageDestroy(sd.vpiImage);
#endif

    if (sd.hostLumaBuffer)
      cuMemFreeHost(sd.hostLumaBuffer);

    if (sd.hostChromaBuffer)
      cuMemFreeHost(sd.hostChromaBuffer);
  }
}

void DebugCameraProvider::updateSurfaces() {
  if (!m_streamFrameReadyToConsume)
    return; // early-out

  pthread_mutex_lock(&m_frameConsumedMutex);

  for (size_t streamIdx = 0; streamIdx < m_streamCount; ++streamIdx) {
    StreamData& sd = m_streamData[streamIdx];

    sd.gpuMatLuma.upload(cvMatLuma(streamIdx));
    sd.gpuMatChroma.upload(cvMatChroma(streamIdx));
  }

  // TODO: need a lower-synchronization-overhead way of handling the uploads
  cuCtxSynchronize();

  // Wake the stream reader thread
  m_streamFrameReadyToConsume = false;
  pthread_cond_signal(&m_frameConsumedCond);
  pthread_mutex_unlock(&m_frameConsumedMutex);

  for (size_t streamIdx = 0; streamIdx < m_streamCount; ++streamIdx) {
    StreamData& sd = m_streamData[streamIdx];

    // Ensure RGB[A] mats are allocated before color conversion
    sd.gpuMatRGBA.create(m_streamHeight, m_streamWidth, CV_8UC4);

    Npp8u const* pSrc[] = {(const Npp8u*) sd.gpuMatLuma.cudaPtr(), (const Npp8u*) sd.gpuMatChroma.cudaPtr()};

    NppiSize sz;
    sz.width = m_streamWidth;
    sz.height = m_streamHeight;
    assert(sd.gpuMatLuma.step == sd.gpuMatChroma.step);

    // No direct NV12 to RGBA conversion function, so we chain NV12 to RGB and then RGB to RGBA
    nppiNV12ToRGB_709HDTV_8u_P2C3R(pSrc, sd.gpuMatLuma.step, (Npp8u*) m_gpuMatRGBTmp.cudaPtr(), m_gpuMatRGBTmp.step, sz);
    int destinationOrder[] = {0, 1, 2, 3};
    nppiSwapChannels_8u_C3C4R((const Npp8u*) m_gpuMatRGBTmp.cudaPtr(), m_gpuMatRGBTmp.step,
      (Npp8u*) sd.gpuMatRGBA.cudaPtr(), sd.gpuMatRGBA.step,
      sz, destinationOrder, /*constant (source channel 3) value=*/ 0xff);

    RHICUDA::copyGpuMatToSurface(sd.gpuMatRGBA, sd.rhiSurfaceRGBA);
  }

  uint32_t srcStereoViewIdx = 0;
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    vd->receivedDisparity.create(m_disparityHeight, m_disparityWidth, CV_16UC1);
    memcpy(vd->receivedDisparity.data, m_stereoDisparityRecvMats[srcStereoViewIdx].data, m_stereoDisparitySizeBytes);

    ++srcStereoViewIdx;
  }
}

cv::Mat DebugCameraProvider::cvMatLuma(size_t streamIdx) const {
  // TODO support more formats
  return cv::Mat(/*rows=*/ m_lumaResourceDescriptor.res.pitch2D.height, /*cols=*/ m_lumaResourceDescriptor.res.pitch2D.width, CV_8U,
    m_streamData[streamIdx].hostLumaBuffer, m_lumaResourceDescriptor.res.pitch2D.pitchInBytes);
}

cv::Mat DebugCameraProvider::cvMatChroma(size_t streamIdx) const {
  // TODO support more formats
  return cv::Mat(/*rows=*/ m_chromaResourceDescriptor.res.pitch2D.height, /*cols=*/ m_chromaResourceDescriptor.res.pitch2D.width, CV_8UC2,
    m_streamData[streamIdx].hostChromaBuffer, m_chromaResourceDescriptor.res.pitch2D.pitchInBytes);
}

cv::cuda::GpuMat DebugCameraProvider::gpuMatGreyscale(size_t streamIdx) {
  return m_streamData[streamIdx].gpuMatLuma;
}

VPIImage DebugCameraProvider::vpiImage(size_t streamIdx) const {
#if HAVE_VPI2
  return m_streamData[streamIdx].vpiImage;
#else
  assert(false && "vpiImageGreyscale: HAVE_VPI2 was not defined at compile time");
  return nullptr;
#endif
}

RHISurface::ptr DebugCameraProvider::rgbTexture(size_t streamIdx) const {
  return m_streamData[streamIdx].rhiSurfaceRGBA;
}

CUtexObject DebugCameraProvider::cudaLumaTexObject(size_t streamIdx) const {
  return m_streamData[streamIdx].cudaLumaTexObject;
}

CUtexObject DebugCameraProvider::cudaChromaTexObject(size_t streamIdx) const {
  return m_streamData[streamIdx].cudaChromaTexObject;
}


// === DepthMapGenerator functions ===


void DebugCameraProvider::internalLoadSettings(cv::FileStorage& fs) {

}

void DebugCameraProvider::internalSaveSettings(cv::FileStorage& fs) {

}


void DebugCameraProvider::internalUpdateViewData() {
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    // CameraSystem::View& v = m_cameraSystem->viewAtIndex(viewIdx);
    auto vd = viewDataAtIndex(viewIdx);

    if (!vd->m_isStereoView)
      continue;

    vd->updateDisparityTexture(internalWidth(), internalHeight(), kSurfaceFormat_R16i);

    vd->receivedDisparity.create(internalHeight(), internalWidth(), CV_16UC1);
  }
}


void DebugCameraProvider::internalProcessFrame() {

  // Fill render surfaces with fake data
  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    if (m_debugDisparityCPUAccessEnabled)
      vd->ensureDebugCPUAccessEnabled(/*bytesPerPixel=*/ 2);

    vd->m_disparityGpuMat.upload(vd->receivedDisparity);

    if (m_debugDisparityCPUAccessEnabled)
      memcpy(vd->m_debugCPUDisparity, vd->receivedDisparity.ptr<uint16_t>(0), sizeof(uint16_t) * vd->receivedDisparity.cols * vd->receivedDisparity.rows);

#if 0
    if (m_populateDebugTextures) {
      if (!vd->m_leftGray)
        vd->m_leftGray = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      if (!vd->m_rightGray)
        vd->m_rightGray = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      RHICUDA::copyGpuMatToSurface(vd->resizedLeft_gpu, vd->m_leftGray, m_globalStream);
      RHICUDA::copyGpuMatToSurface(vd->resizedRight_gpu, vd->m_rightGray, m_globalStream);
    }
#endif
  }

  internalFinalizeDisparityTexture();
}

void DebugCameraProvider::internalRenderIMGUI() {
}

void DebugCameraProvider::internalRenderIMGUIPerformanceGraphs() {
}

