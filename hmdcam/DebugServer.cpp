#include "DebugServer.h"
#include "common/CameraSystem.h"
#include "common/DepthMapGenerator.h"
#include "IArgusCamera.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <errno.h>

#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/CudaUtil.h"

const int kRetryDelaySeconds = 10;

#define die(msg, ...)                         \
  do {                                        \
    fprintf(stderr, msg "\n", ##__VA_ARGS__); \
    abort();                                  \
  } while (0)
#define TEST_NN_ERRNO(x)                                                         \
  do {                                                                           \
    if (((x) < 0)) die("error: " #x " failed: %d (%s)", errno, strerror(errno)); \
  } while (0)

#define TEST_NZ(x)                                             \
  do {                                                         \
    if ((x)) die("error: " #x " failed (returned non-zero)."); \
  } while (0)
#define TEST_Z(x)                                                \
  do {                                                           \
    if (!(x)) die("error: " #x " failed (returned zero/null)."); \
  } while (0)

DebugServer::DebugServer() {
  memset(&m_lumaResourceDescriptor, 0, sizeof(m_lumaResourceDescriptor));
  memset(&m_chromaResourceDescriptor, 0, sizeof(m_chromaResourceDescriptor));
}

DebugServer::~DebugServer() {
  if (m_streamThread) {
    pthread_cancel(m_streamThread);
    pthread_join(m_streamThread, NULL);
  }
  delete[] m_streamResources;
}


bool DebugServer::initWithCameraSystem(CameraSystem* cs, IArgusCamera* cp, DepthMapGenerator* depth) {
  m_cameraSystem = cs;
  m_cameraProvider = cp;
  m_depthMapGenerator = depth;

  m_streamCount = m_cameraProvider->streamCount();
  m_streamResources = new StreamResource[m_streamCount];

  // Check resource types, allocate StreamResource structs and host-side luma/chroma planes

  // We assume that all streams have identical buffer types

  {
    CUtexObject lumaTex = m_cameraProvider->cudaLumaTexObject(0);
    CUtexObject chromaTex = m_cameraProvider->cudaChromaTexObject(0);
    if (!lumaTex || !chromaTex) {
      printf("CameraProvider failed to provide luma or chroma CUtexObject for stream 0\n");
      return false;
    }

    // Fill resource descriptors from the CUtexObjects.
    // (we're just going to serialize the contents of these descriptors to populate the config buffer.)
    CUDA_CHECK(cuTexObjectGetResourceDesc(&m_lumaResourceDescriptor, lumaTex));
    CUDA_CHECK(cuTexObjectGetResourceDesc(&m_chromaResourceDescriptor, chromaTex));

    // We pull the data from the pitch2d slot in the resource descriptor.
    // TODO: It'd be pretty easy to support arrays, also, via cuArrayGetDescriptor
    assert(m_lumaResourceDescriptor.resType == CU_RESOURCE_TYPE_PITCH2D);
    assert(m_chromaResourceDescriptor.resType == CU_RESOURCE_TYPE_PITCH2D);

    printf("Stream   Luma: %zu x %zu NumChannels=%u Format=0x%x pitchInBytes=%zu\n",
      m_lumaResourceDescriptor.res.pitch2D.width, m_lumaResourceDescriptor.res.pitch2D.height,
      m_lumaResourceDescriptor.res.pitch2D.numChannels,
      m_lumaResourceDescriptor.res.pitch2D.format,
      m_lumaResourceDescriptor.res.pitch2D.pitchInBytes);
    printf("Stream Chroma: %zu x %zu NumChannels=%u Format=0x%x pitchInBytes=%zu\n",
      m_chromaResourceDescriptor.res.pitch2D.width, m_chromaResourceDescriptor.res.pitch2D.height,
      m_chromaResourceDescriptor.res.pitch2D.numChannels,
      m_chromaResourceDescriptor.res.pitch2D.format,
      m_chromaResourceDescriptor.res.pitch2D.pitchInBytes);

    // TODO handle other type-sizes
    assert(m_lumaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_UNSIGNED_INT8 || m_lumaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_SIGNED_INT8);
    assert(m_chromaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_UNSIGNED_INT8 || m_chromaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_SIGNED_INT8);

    m_lumaPlaneSizeBytes = m_lumaResourceDescriptor.res.pitch2D.height * m_lumaResourceDescriptor.res.pitch2D.pitchInBytes;
    m_chromaPlaneSizeBytes = m_chromaResourceDescriptor.res.pitch2D.height * m_chromaResourceDescriptor.res.pitch2D.pitchInBytes;
  }

  // Allocate host buffers for chroma/luma plane copies
  for (size_t cameraIdx = 0; cameraIdx < m_streamCount; ++cameraIdx) {
    CUDA_CHECK(cuMemHostAlloc(&m_streamResources[cameraIdx].m_lumaPlane, m_lumaPlaneSizeBytes, /*flags=*/ 0));
    CUDA_CHECK(cuMemHostAlloc(&m_streamResources[cameraIdx].m_chromaPlane, m_chromaPlaneSizeBytes, /*flags=*/ 0));
  }

  // Serialize config information for the client
  SerializationBuffer cfg;
  cfg.put_u32(m_cameraProvider->streamCount());
  cfg.put_u32(m_cameraProvider->streamWidth());
  cfg.put_u32(m_cameraProvider->streamHeight());

  cfg.put_u32(m_lumaResourceDescriptor.res.pitch2D.format);
  cfg.put_u32(m_lumaResourceDescriptor.res.pitch2D.numChannels);
  cfg.put_u32(m_lumaResourceDescriptor.res.pitch2D.width);
  cfg.put_u32(m_lumaResourceDescriptor.res.pitch2D.height);
  cfg.put_u32(m_lumaResourceDescriptor.res.pitch2D.pitchInBytes);

  cfg.put_u32(m_chromaResourceDescriptor.res.pitch2D.format);
  cfg.put_u32(m_chromaResourceDescriptor.res.pitch2D.numChannels);
  cfg.put_u32(m_chromaResourceDescriptor.res.pitch2D.width);
  cfg.put_u32(m_chromaResourceDescriptor.res.pitch2D.height);
  cfg.put_u32(m_chromaResourceDescriptor.res.pitch2D.pitchInBytes);

  // Depth map generator data
  uint32_t stereoViews = 0;
  if (m_depthMapGenerator) {
    // Count stereo views
    for (size_t viewIdx = 0; viewIdx < cs->views(); ++viewIdx) {
      const CameraSystem::View& v = cs->viewAtIndex(viewIdx);
      if (v.isStereo)
        ++stereoViews;
    }
  }

  cfg.put_u32(stereoViews); // Stereo view count, or 0 if we don't have a depth map generator
  if (stereoViews) {
    cfg.put_u32(m_depthMapGenerator->algoInputWidth());
    cfg.put_u32(m_depthMapGenerator->algoInputHeight());

    cfg.put_u32(m_depthMapGenerator->internalWidth());
    cfg.put_u32(m_depthMapGenerator->internalHeight());

    cfg.put_u32(m_depthMapGenerator->m_algoDownsampleX);
    cfg.put_u32(m_depthMapGenerator->m_algoDownsampleY);

    cfg.put_u32(m_depthMapGenerator->m_maxDisparityPixels);
    cfg.put_u32(m_depthMapGenerator->m_disparitySubpixelBits);

    m_disparityInputStreamSizeBytes = m_depthMapGenerator->algoInputWidth() * m_depthMapGenerator->algoInputHeight() * sizeof(uint8_t);
    for (uint32_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
      m_disparityInputStreams[eyeIdx].resize(stereoViews);
      for (uint32_t viewIdx = 0; viewIdx < stereoViews; ++viewIdx) {
        m_disparityInputStreams[eyeIdx][viewIdx].create(m_depthMapGenerator->algoInputHeight(), m_depthMapGenerator->algoInputWidth(), CV_8U);
      }
    }

    m_disparityStreamSizeBytes = m_depthMapGenerator->internalWidth() * m_depthMapGenerator->internalHeight() * sizeof(uint16_t);
    m_disparityStreams.resize(stereoViews);
    for (uint32_t i = 0; i < stereoViews; ++i) {
      m_disparityStreams[i].create(m_depthMapGenerator->internalHeight(), m_depthMapGenerator->internalWidth(), CV_16UC1);
    }

    m_disparityDebugResidualStreamSizeBytes = m_depthMapGenerator->internalWidth() * m_depthMapGenerator->internalHeight() * sizeof(uint8_t);
    m_disparityDebugResidualStreams.resize(stereoViews);
    for (uint32_t i = 0; i < stereoViews; ++i) {
      m_disparityDebugResidualStreams[i].create(m_depthMapGenerator->internalHeight(), m_depthMapGenerator->internalWidth(), CV_8U);
    }
  }

  m_streamHeader = cfg;

  {
    // Serialize the CameraSystem config
    cv::FileStorage fs(cv::String(), cv::FileStorage::MEMORY | cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    cs->saveCalibrationData(fs);
    m_cameraSystemConfig = fs.releaseAndGetString();
  }

  // Start the listener thread
  pthread_create(&m_streamThread, NULL, &streamThreadEntryPoint, (void*) this);
  return true;
}

void DebugServer::frameProcessingEnded() {
  // Turn on CPU access to disparity only if we have a client connected
  // (this can be freely enabled/disabled without reconfiguring the depth backend,
  // and will take effect next frame)
  if (m_depthMapGenerator)
    m_depthMapGenerator->setDebugDisparityCPUAccessEnabled(m_streamConnected);

  if (!m_streamConnected)
    return; // Don't bother doing any work if we don't have a client

  if (!m_streamReadyForNextFrame)
    return; // Probably still writing out the last frame

  // Copy luma/chroma planes to stream resources
  for (size_t cameraIdx = 0; cameraIdx < m_cameraProvider->streamCount(); ++cameraIdx) {
    { // Luma copy
      CUDA_MEMCPY2D copyDescriptor;
      memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));

      m_cameraProvider->fillCudaMemcpy2DForStreamSource(copyDescriptor, cameraIdx, /*fromChromaPlane= */ false);

      copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
      copyDescriptor.dstHost = m_streamResources[cameraIdx].m_lumaPlane;
      copyDescriptor.dstPitch = m_lumaResourceDescriptor.res.pitch2D.pitchInBytes;

      CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
    }

    { // Chroma copy
      CUDA_MEMCPY2D copyDescriptor;
      memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));

      m_cameraProvider->fillCudaMemcpy2DForStreamSource(copyDescriptor, cameraIdx, /*fromChromaPlane= */ true);

      copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
      copyDescriptor.dstHost = m_streamResources[cameraIdx].m_chromaPlane;
      copyDescriptor.dstPitch = m_chromaResourceDescriptor.res.pitch2D.pitchInBytes;

      CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
    }
  }

  // Copy disparity (sourced from the CPU-accessible debug view functionality)
  if (m_depthMapGenerator) {
    uint32_t dispStreamIdx = 0;
    for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
      auto vd = m_depthMapGenerator->viewDataAtIndex(viewIdx);
      if (!vd->m_isStereoView)
        continue;

      if (vd->m_debugCPUDisparity.empty() || vd->m_debugCPUDisparityInput[0].empty() || vd->m_debugCPUDisparityInput[1].empty())
        continue;

      for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)
        memcpy(m_disparityInputStreams[eyeIdx][dispStreamIdx].data, vd->m_debugCPUDisparityInput[eyeIdx].ptr(), m_disparityInputStreamSizeBytes);

      memcpy(m_disparityStreams[dispStreamIdx].data, vd->m_debugCPUDisparity.ptr(), m_disparityStreamSizeBytes);

      memcpy(m_disparityDebugResidualStreams[dispStreamIdx].data, vd->m_debugCPUDisparityResidual.ptr(), m_disparityDebugResidualStreamSizeBytes);

      ++dispStreamIdx;
    }
  }

  pthread_mutex_lock(&m_streamReadyMutex);
  m_streamReadyForNextFrame = false;
  // Wake the stream thread
  pthread_cond_signal(&m_streamReadyCond);
  pthread_mutex_unlock(&m_streamReadyMutex);
}

bool safe_write(int fd, const void* buffer, size_t length) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(buffer);
  size_t remaining = length;
  while (remaining) {
    ssize_t res = write(fd, p, remaining);
    if (res < 0)
      return false;

    p += res;
    remaining -= res;
  }
  return true;
}

void DebugServer::streamThreadFn() {
  pthread_setname_np(pthread_self(), "DebugServer_Stream");

  int listenFd;
  TEST_NN_ERRNO(listenFd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));

  struct sockaddr_in listenAddr;
  memset(&listenAddr, '0', sizeof(listenAddr));

  listenAddr.sin_family = AF_INET;
  listenAddr.sin_addr.s_addr = htonl(INADDR_ANY);
  listenAddr.sin_port = htons(55443);

  while (bind(listenFd, (struct sockaddr*) &listenAddr, sizeof(listenAddr)) < 0) {
    if (errno == EADDRINUSE) {
      fprintf(stderr, "bind(): address in use, retrying in %ds\n", kRetryDelaySeconds);
      sleep(kRetryDelaySeconds);
      continue;
    }
    die("bind(): failed: %s", strerror(errno));
  }

  TEST_NN_ERRNO(listen(listenFd, 1));

  // close listening socket on thread termination
  pthread_cleanup_push((void (*)(void*)) close, (void*) static_cast<ssize_t>(listenFd));

  while (true) { // connection loop
    struct sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    int clientFd = accept4(listenFd, (struct sockaddr*) &clientAddr, &clientAddrLen, SOCK_CLOEXEC);
    if (clientFd < 0) {
      perror("accept");
      continue;
    }

    // close client socket on thread termination
    pthread_cleanup_push((void (*)(void*)) close, (void*) static_cast<ssize_t>(clientFd));
    m_streamConnected = true;


    {
      // Write stream data header
      uint32_t streamHeaderSize = boost::endian::native_to_big<uint32_t>(m_streamHeader.size());
      if (!safe_write(clientFd, &streamHeaderSize, sizeof(streamHeaderSize)))
        goto cleanup;

      if (!safe_write(clientFd, m_streamHeader.data(), m_streamHeader.size()))
        goto cleanup;

      uint32_t csConfigSize = boost::endian::native_to_big<uint32_t>(m_cameraSystemConfig.size());
      if (!safe_write(clientFd, &csConfigSize, sizeof(csConfigSize)))
        goto cleanup;

      if (!safe_write(clientFd, m_cameraSystemConfig.data(), m_cameraSystemConfig.size()))
        goto cleanup;

      while (true) {
        // Sync with main thread -- signal that we're waiting on a frame, and wait for it to be copied into the StreamResource buffers
        pthread_mutex_lock(&m_streamReadyMutex);
        m_streamReadyForNextFrame = true;
        pthread_cond_wait(&m_streamReadyCond, &m_streamReadyMutex);
        m_streamReadyForNextFrame = false;
        pthread_mutex_unlock(&m_streamReadyMutex);

        for (uint32_t streamIdx = 0; streamIdx < m_streamCount; ++streamIdx) {
          if (!safe_write(clientFd, m_streamResources[streamIdx].m_lumaPlane, m_lumaPlaneSizeBytes)) goto cleanup;
          if (!safe_write(clientFd, m_streamResources[streamIdx].m_chromaPlane, m_chromaPlaneSizeBytes)) goto cleanup;
        }
        for (size_t dispStreamIdx = 0; dispStreamIdx < m_disparityStreams.size(); ++dispStreamIdx) {
          if (!safe_write(clientFd, m_disparityInputStreams[0][dispStreamIdx].data, m_disparityInputStreamSizeBytes)) goto cleanup;
          if (!safe_write(clientFd, m_disparityInputStreams[1][dispStreamIdx].data, m_disparityInputStreamSizeBytes)) goto cleanup;
          if (!safe_write(clientFd, m_disparityStreams[dispStreamIdx].data, m_disparityStreamSizeBytes)) goto cleanup;
          if (!safe_write(clientFd, m_disparityDebugResidualStreams[dispStreamIdx].data, m_disparityDebugResidualStreamSizeBytes)) goto cleanup;
        }
      } // frame loop
    }

  cleanup:
    m_streamConnected = false;
    pthread_cleanup_pop(/*execute=*/ 1); // close clientFd
  }
  pthread_cleanup_pop(/*execute=*/ 1); // close listenFd
}
