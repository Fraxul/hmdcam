#include "DebugServer.h"
#include "common/CameraSystem.h"
#include "IArgusCamera.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <errno.h>

#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/gl/GLCommon.h" // must be included before cudaEGL

#if !defined(IS_WSL2) // CUDA-EGL interop doesn't exist on WSL2
#include <cudaEGL.h>
#include <cuda_egl_interop.h>
#endif

const int kRetryDelaySeconds = 10;

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)
#define TEST_NN_ERRNO(x) do { if ( ((x) < 0)) die("error: " #x " failed: %d (%s)", errno, strerror(errno) ); } while (0)

#define TEST_NZ(x) do { if ( (x)) die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) die("error: " #x " failed (returned zero/null)."); } while (0)

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


bool DebugServer::initWithCameraSystem(CameraSystem* cs, IArgusCamera* cp) {
#ifdef IS_WSL2
  printf("DebugServer::initWithCameraSystem(): Built with IS_WSL2, debug server support is disabled due to missing APIs.\n");
  return false;
#else

  m_cameraSystem = cs;
  m_cameraProvider = cp;

  CUeglColorFormat eglColorFormat;

  m_streamCount = m_cameraProvider->streamCount();
  m_streamResources = new StreamResource[m_streamCount];

  // Check resource types, allocate StreamResource structs and host-side luma/chroma planes

  for (size_t cameraIdx = 0; cameraIdx < m_streamCount; ++cameraIdx) {
    CUgraphicsResource rsrc = m_cameraProvider->cudaGraphicsResource(cameraIdx);
    if (!rsrc) {
      printf("CameraProvider failed to provide CUgraphicsResource for stream %zu\n", cameraIdx);
      return false;
    }

    // Using the Runtime API here instead since it gives better information about multiplanar formats
    cudaEglFrame eglFrame;
    CUDA_CHECK(cudaGraphicsResourceGetMappedEglFrame(&eglFrame, (cudaGraphicsResource_t) rsrc, /*cubemapIndex=*/ 0, /*mipLevel=*/ 0));
    assert(eglFrame.frameType == cudaEglFrameTypePitch);
    assert(eglFrame.planeCount == 2);

    if (cameraIdx == 0) {
      eglColorFormat = (CUeglColorFormat) eglFrame.eglColorFormat; // CUeglColorFormat and cudaEglColorFormat are interchangeable

      // Convert eglFrame to resource descriptors.
      // We don't fill the device pointers, we're just going to serialize the contents
      // of these descriptors to populate the config buffer.
      m_lumaResourceDescriptor.resType = CU_RESOURCE_TYPE_PITCH2D;
      m_lumaResourceDescriptor.res.pitch2D.devPtr       = 0;
      m_lumaResourceDescriptor.res.pitch2D.format       = CU_AD_FORMAT_UNSIGNED_INT8; // eglFrame.planeDesc[0].channelDesc.f; // TODO
      m_lumaResourceDescriptor.res.pitch2D.numChannels  = eglFrame.planeDesc[0].numChannels;
      m_lumaResourceDescriptor.res.pitch2D.width        = eglFrame.planeDesc[0].width;
      m_lumaResourceDescriptor.res.pitch2D.height       = eglFrame.planeDesc[0].height;
      m_lumaResourceDescriptor.res.pitch2D.pitchInBytes = eglFrame.planeDesc[0].pitch;

      // TODO hardcoded assumptions about chroma format -- we should be able to get this from the eglColorFormat!
      m_chromaResourceDescriptor.res.pitch2D.devPtr       = 0;
      m_chromaResourceDescriptor.res.pitch2D.format       = CU_AD_FORMAT_UNSIGNED_INT8; // eglFrame.planeDesc[1].channelDesc.f // TODO
      m_chromaResourceDescriptor.res.pitch2D.numChannels  = 2; // eglFrame.planeDesc[1].numChannels; // TODO
      m_chromaResourceDescriptor.res.pitch2D.width        = eglFrame.planeDesc[1].width;
      m_chromaResourceDescriptor.res.pitch2D.height       = eglFrame.planeDesc[1].height;
      // pitchInBytes NOTE: "...in case of multiplanar *eglFrame, pitch of only first plane is to be considered by the application."
      // (accessing planeDesc[0] is intentional)
      m_chromaResourceDescriptor.res.pitch2D.pitchInBytes = eglFrame.planeDesc[0].pitch;

      printf("Stream [%zu]:   Luma: %zu x %zu NumChannels=%u ChannelDesc=0x%x (%d,%d,%d,%d) pitchInBytes=%zu\n", cameraIdx,
        m_lumaResourceDescriptor.res.pitch2D.width, m_lumaResourceDescriptor.res.pitch2D.height,
        m_lumaResourceDescriptor.res.pitch2D.numChannels,
        eglFrame.planeDesc[0].channelDesc.f, eglFrame.planeDesc[0].channelDesc.x, eglFrame.planeDesc[0].channelDesc.y,
        eglFrame.planeDesc[0].channelDesc.z, eglFrame.planeDesc[0].channelDesc.w,
        m_lumaResourceDescriptor.res.pitch2D.pitchInBytes);
      printf("Stream [%zu]: Chroma: %zu x %zu NumChannels=%u ChannelDesc=0x%x (%d,%d,%d,%d) pitchInBytes=%zu\n", cameraIdx,
        m_chromaResourceDescriptor.res.pitch2D.width, m_chromaResourceDescriptor.res.pitch2D.height,
        m_chromaResourceDescriptor.res.pitch2D.numChannels,
        eglFrame.planeDesc[1].channelDesc.f, eglFrame.planeDesc[1].channelDesc.x, eglFrame.planeDesc[1].channelDesc.y,
        eglFrame.planeDesc[1].channelDesc.z, eglFrame.planeDesc[1].channelDesc.w,
        m_chromaResourceDescriptor.res.pitch2D.pitchInBytes);

      // TODO handle other type-sizes
      assert(m_lumaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_UNSIGNED_INT8 || m_lumaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_SIGNED_INT8);
      assert(m_chromaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_UNSIGNED_INT8 || m_chromaResourceDescriptor.res.pitch2D.format == CU_AD_FORMAT_SIGNED_INT8);

      m_lumaPlaneSizeBytes = m_lumaResourceDescriptor.res.pitch2D.height * m_lumaResourceDescriptor.res.pitch2D.pitchInBytes;
      m_chromaPlaneSizeBytes = m_chromaResourceDescriptor.res.pitch2D.height * m_chromaResourceDescriptor.res.pitch2D.pitchInBytes;
    }


    CUDA_CHECK(cuMemHostAlloc(&m_streamResources[cameraIdx].m_lumaPlane, m_lumaPlaneSizeBytes, /*flags=*/ 0));
    CUDA_CHECK(cuMemHostAlloc(&m_streamResources[cameraIdx].m_chromaPlane, m_chromaPlaneSizeBytes, /*flags=*/ 0));
  }

  // Serialize config information for the client
  SerializationBuffer cfg;
  cfg.put_u32(m_cameraProvider->streamCount());
  cfg.put_u32(m_cameraProvider->streamWidth());
  cfg.put_u32(m_cameraProvider->streamHeight());

  cfg.put_u32(eglColorFormat);

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

  m_streamHeader = cfg;

  // Start the listener thread
  pthread_create(&m_streamThread, NULL, &streamThreadEntryPoint, (void*) this);
  return true;
#endif // IS_WSL2
}

void DebugServer::frameProcessingEnded() {
  if (!m_streamConnected)
    return; // Don't bother doing any work if we don't have a client

  if (!m_streamReadyForNextFrame)
    return; // Probably still writing out the last frame

  // Copy luma/chroma planes to stream resources
  for (size_t cameraIdx = 0; cameraIdx < m_cameraProvider->streamCount(); ++cameraIdx) {
    CUeglFrame eglFrame;
    CUDA_CHECK(cuGraphicsResourceGetMappedEglFrame(&eglFrame, m_cameraProvider->cudaGraphicsResource(cameraIdx), 0, 0));

    { // Luma copy
      CUDA_MEMCPY2D copyDescriptor;
      memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));

      copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      copyDescriptor.srcDevice = (CUdeviceptr) eglFrame.frame.pPitch[0];
      copyDescriptor.srcPitch = m_lumaResourceDescriptor.res.pitch2D.pitchInBytes;

      copyDescriptor.WidthInBytes = m_lumaResourceDescriptor.res.pitch2D.width * 1; //8-bit, 1-channel
      copyDescriptor.Height = m_lumaResourceDescriptor.res.pitch2D.height;

      copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
      copyDescriptor.dstHost = m_streamResources[cameraIdx].m_lumaPlane;
      copyDescriptor.dstPitch = m_lumaResourceDescriptor.res.pitch2D.pitchInBytes;

      CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
    }

    { // Chroma copy
      CUDA_MEMCPY2D copyDescriptor;
      memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));

      copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
      copyDescriptor.srcDevice = (CUdeviceptr) eglFrame.frame.pPitch[1];
      copyDescriptor.srcPitch = m_chromaResourceDescriptor.res.pitch2D.pitchInBytes;

      copyDescriptor.WidthInBytes = m_chromaResourceDescriptor.res.pitch2D.width * 2; //8-bit, 2-channel
      copyDescriptor.Height = m_chromaResourceDescriptor.res.pitch2D.height;

      copyDescriptor.dstMemoryType = CU_MEMORYTYPE_HOST;
      copyDescriptor.dstHost = m_streamResources[cameraIdx].m_chromaPlane;
      copyDescriptor.dstPitch = m_chromaResourceDescriptor.res.pitch2D.pitchInBytes;

      CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
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

  while (bind(listenFd, (struct sockaddr*)&listenAddr, sizeof(listenAddr)) < 0) {
    if (errno == EADDRINUSE) {
      fprintf(stderr, "bind(): address in use, retrying in %ds\n", kRetryDelaySeconds);
      sleep(kRetryDelaySeconds);
      continue;
    }
    die("bind(): failed: %s", strerror(errno));
  }

  TEST_NN_ERRNO(listen(listenFd, 1));

  // close listening socket on thread termination
  pthread_cleanup_push((void(*)(void*)) close, (void*) static_cast<ssize_t>(listenFd));

  while (true) { // connection loop
    struct sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    int clientFd = accept4(listenFd, (struct sockaddr*)&clientAddr, &clientAddrLen, SOCK_CLOEXEC);
    if (clientFd < 0) {
      perror("accept");
      continue;
    }

    // close client socket on thread termination
    pthread_cleanup_push((void(*)(void*)) close, (void*) static_cast<ssize_t>(clientFd));
    m_streamConnected = true;


    // Write stream data header
    uint32_t streamHeaderSize = boost::endian::native_to_big<uint32_t>(m_streamHeader.size());
    if (!safe_write(clientFd, &streamHeaderSize, sizeof(streamHeaderSize)))
      goto cleanup;

    if (!safe_write(clientFd, m_streamHeader.data(), m_streamHeader.size()))
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
    } // frame loop

cleanup:
    m_streamConnected = false;
    pthread_cleanup_pop(/*execute=*/ 1); // close clientFd
  }
  pthread_cleanup_pop(/*execute=*/ 1); // close listenFd
}


