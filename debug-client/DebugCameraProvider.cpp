#include "DebugCameraProvider.h"
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/Timing.h"
#include "common/glmCvInterop.h"
#include "common/remapArray.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/cuda/RHICUDA.h"
#include "rhi/cuda/RHICVInterop.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <string>
#include <assert.h>
#include <cuda.h>
#include <npp.h>

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

DebugCameraProvider::DebugCameraProvider() :
  DepthMapGenerator(kDepthBackendMock) {
}

bool DebugCameraProvider::connect(const char* debugHost) {
  if (m_fd >= 0) {
    close(m_fd);
    m_fd = -1;
  }

  // Resolve host / establish connection
  {
    struct addrinfo hints;
    struct addrinfo* result = nullptr;
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
    m_recvAlgoInputWidth = cfg.get_u32();
    m_recvAlgoInputHeight = cfg.get_u32();

    m_disparityWidth = cfg.get_u32();
    m_disparityHeight = cfg.get_u32();

    m_algoDownsampleX = cfg.get_u32();
    m_algoDownsampleY = cfg.get_u32();

    m_maxDisparityPixels = cfg.get_u32();
    m_disparitySubpixelBits = cfg.get_u32();

    m_stereoDisparityInputSizeBytes = m_recvAlgoInputWidth * m_recvAlgoInputHeight * sizeof(uint8_t);
    m_stereoDisparitySizeBytes = m_disparityWidth * m_disparityHeight * sizeof(uint16_t);
    m_stereoDisparityDebugResidualSizeBytes = m_disparityWidth * m_disparityHeight * sizeof(uint8_t);

    printf("Algo: %ux%u Disp: %ux%u Downsample: %ux%u maxDisp=%u subpixelBits=%u inputSizeBytes=%u dispSizeBytes=%u residualSizeBytes=%u\n",
      m_recvAlgoInputWidth, m_recvAlgoInputHeight, m_disparityWidth, m_disparityHeight, m_algoDownsampleX, m_algoDownsampleY,
      m_maxDisparityPixels, m_disparitySubpixelBits, m_stereoDisparityInputSizeBytes, m_stereoDisparitySizeBytes, m_stereoDisparityDebugResidualSizeBytes);

    for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
      m_stereoDisparityInputRecvMats[eyeIdx].resize(m_stereoViewCount);
      for (uint32_t viewIdx = 0; viewIdx < m_stereoViewCount; ++viewIdx) {
        m_stereoDisparityInputRecvMats[eyeIdx][viewIdx].create(m_recvAlgoInputHeight, m_recvAlgoInputWidth, CV_8U);
      }
    }

    m_stereoDisparityRecvMats.resize(m_stereoViewCount);
    for (uint32_t viewIdx = 0; viewIdx < m_stereoViewCount; ++viewIdx) {
      m_stereoDisparityRecvMats[viewIdx].create(m_disparityHeight, m_disparityWidth, CV_16UC1);
    }

    m_stereoDisparityDebugResidualRecvMats.resize(m_stereoViewCount);
    for (uint32_t viewIdx = 0; viewIdx < m_stereoViewCount; ++viewIdx) {
      m_stereoDisparityDebugResidualRecvMats[viewIdx].create(m_disparityHeight, m_disparityWidth, CV_8U);
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

    // Allocate as an RHI interop surface so we can copy from CUDA GpuMat without round-tripping through host memory.
    sd.rhiSurfaceRGBA = rhi()->newInteropSurface(streamWidth(), streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
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
      texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
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
      texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
      texDesc.maxAnisotropy = 1;

      CUDA_CHECK(cuTexObjectCreate(&sd.cudaChromaTexObject, &m_chromaResourceDescriptor, &texDesc, /*resourceViewDescriptor=*/ nullptr));
    }
  }

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
      if (!safe_read(m_fd, m_stereoDisparityInputRecvMats[0][stereoViewIdx].data, m_stereoDisparityInputSizeBytes)) goto cleanup;
      if (!safe_read(m_fd, m_stereoDisparityInputRecvMats[1][stereoViewIdx].data, m_stereoDisparityInputSizeBytes)) goto cleanup;
      if (!safe_read(m_fd, m_stereoDisparityRecvMats[stereoViewIdx].data, m_stereoDisparitySizeBytes)) goto cleanup;
      if (!safe_read(m_fd, m_stereoDisparityDebugResidualRecvMats[stereoViewIdx].data, m_stereoDisparityDebugResidualSizeBytes)) goto cleanup;
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

    // Async D2D copy on the default stream; the global signalCUDAToRHI at
    // the end of the frame extends the timeline past this work, so the
    // following VK render submit's wait covers it.
    RHICUDA::copyGpuMatToSurface(sd.gpuMatRGBA, sd.rhiSurfaceRGBA, RHICUDA::defaultAsyncStream);
  }

  uint32_t srcStereoViewIdx = 0;
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    memcpy(vd->receivedDisparityInput[0].data, m_stereoDisparityInputRecvMats[0][srcStereoViewIdx].data, m_stereoDisparityInputSizeBytes);
    memcpy(vd->receivedDisparityInput[1].data, m_stereoDisparityInputRecvMats[1][srcStereoViewIdx].data, m_stereoDisparityInputSizeBytes);
    memcpy(vd->receivedDisparity.data, m_stereoDisparityRecvMats[srcStereoViewIdx].data, m_stereoDisparitySizeBytes);
    memcpy(vd->receivedDisparityDebugResidual.data, m_stereoDisparityDebugResidualRecvMats[srcStereoViewIdx].data, m_stereoDisparityDebugResidualSizeBytes);

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

bool DebugCameraProvider::fillCudaMemcpy2DForStreamSource(CUDA_MEMCPY2D& outCopyDescriptor, size_t sensorIndex, bool fromChromaPlane) const {
  if (sensorIndex >= m_streamData.size())
    return false;

  const cv::cuda::GpuMat* src = fromChromaPlane ? &m_streamData[sensorIndex].gpuMatChroma : &m_streamData[sensorIndex].gpuMatLuma;

  if (src->empty())
    return false;

  outCopyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  outCopyDescriptor.srcDevice = (CUdeviceptr) src->ptr<const uint8_t>();
  outCopyDescriptor.srcPitch = src->step;

  outCopyDescriptor.WidthInBytes = src->cols * src->elemSize1();
  outCopyDescriptor.Height = src->rows;
  return true;
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

    vd->updateDisparityTexture(this, internalWidth(), internalHeight(), kSurfaceFormat_R16i);

    // Create receive Mats for the disparity inputs (algoInput size)
    vd->receivedDisparityInput[0].create(m_recvAlgoInputHeight, m_recvAlgoInputWidth, CV_8U);
    vd->receivedDisparityInput[1].create(m_recvAlgoInputHeight, m_recvAlgoInputWidth, CV_8U);

    // Create receive Mats for the disparity outputs (internal size)
    vd->receivedDisparity.create(m_disparityHeight, m_disparityWidth, CV_16UC1);
    vd->receivedDisparityDebugResidual.create(m_disparityHeight, m_disparityWidth, CV_8U);

    // These can directly alias the receivedDisparityInput to save a copy
    vd->m_debugCPUDisparityInput[0] = vd->receivedDisparityInput[0];
    vd->m_debugCPUDisparityInput[1] = vd->receivedDisparityInput[1];
  }
}


void DebugCameraProvider::internalProcessFrame() {

  // Fill render surfaces with received data
  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    vd->currentDisparityMat().upload(vd->receivedDisparity);

    if (m_populateDebugTextures) {
      // TODO: These are interop textures for consistency, but we don't write to them from CUDA.
      // Maybe we should switch from using loadTextureData to doing a cuda memcpy into the CUDA side
      // to follow the data path of the other backends.

      if (!vd->m_leftGray)
        vd->m_leftGray = rhi()->newInteropSurface(algoInputWidth(), algoInputHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      if (!vd->m_rightGray)
        vd->m_rightGray = rhi()->newInteropSurface(algoInputWidth(), algoInputHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      rhi()->loadTextureData(vd->m_leftGray, kVertexElementTypeUByte1N, vd->receivedDisparityInput[0].ptr());
      rhi()->loadTextureData(vd->m_rightGray, kVertexElementTypeUByte1N, vd->receivedDisparityInput[1].ptr());
    }
  }

  internalFinalizeDisparityTexture();
}

void DebugCameraProvider::internalRenderIMGUI() {
}

void DebugCameraProvider::internalRenderIMGUIPerformanceGraphs() {
}

void DebugCameraProvider::internalPostInitWithCameraSystem() {
  // Algorithm dimensions are supposed to be computed in initWithCameraSystem.
  // Update them with the values we received.
  m_algoInputWidth = m_recvAlgoInputWidth;
  m_algoInputHeight = m_recvAlgoInputHeight;
}
