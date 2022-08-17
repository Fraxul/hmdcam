#include "NvEncSession.h"
#include "Render.h"
#include "RenderBackend.h"
#include "rhi/RHI.h"
#include "rhi/gl/RHISurfaceGL.h"
#include "rhi/cuda/CudaUtil.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>
#include <cudaGL.h>
#include <linux/videodev2.h>
#include <linux/v4l2-controls.h>
#include <libv4l2.h>
#include <stdio.h>
#include <string.h>
#include <sys/prctl.h>

#include "nvbuf_utils.h"
#include "nvbufsurface.h"
#include "nvbufsurftransform.h"
#include "NvLogging.h"
#include "NvVideoEncoder.h"
#include "NvUtils.h"

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)
#define CHECK_ZERO(x) if ((x) != 0) { fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); abort(); }
#define CHECK_TRUE(x) if (!(x)) { fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); abort(); }
#define CHECK_NOT_NULL(x) if ((x) == NULL) { fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); abort(); }

static const uint32_t kInputBufferCount = 6;
static const uint32_t kOutputBufferCount = 10;

NvEncSession::NvEncSession(uint32_t _width, uint32_t _height) :
  m_width(_width), m_height(_height),
  m_encoderPixfmt(V4L2_PIX_FMT_H264) {

  // NvLogging
  // log_level = LOG_LEVEL_DEBUG;

  pthread_mutex_init(&m_stateLock, NULL);
  pthread_mutex_init(&m_callbackLock, NULL);

  pthread_mutex_init(&m_gpuSubmissionQueueLock, NULL);
  pthread_cond_init(&m_gpuSubmissionQueueCond, NULL);


  // Allocate surface pool
  NvBufSurfaceAllocateParams vicInputSurfaceParams;
  memset(&vicInputSurfaceParams, 0, sizeof(vicInputSurfaceParams));
  vicInputSurfaceParams.params.width = m_width;
  vicInputSurfaceParams.params.height = m_height;
  vicInputSurfaceParams.params.layout = NVBUF_LAYOUT_PITCH;
  vicInputSurfaceParams.params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  vicInputSurfaceParams.params.memType = NVBUF_MEM_SURFACE_ARRAY;
  vicInputSurfaceParams.memtag = NvBufSurfaceTag_VIDEO_CONVERT;

  m_vicInputSurfaces.resize(kInputBufferCount);
  for (size_t i = 0; i < kInputBufferCount; ++i) {
#if L4T_RELEASE < 34
    // use old API
    CHECK_ZERO(NvBufSurfaceCreate(&m_vicInputSurfaces[i], 1, &vicInputSurfaceParams.params));
#else
    CHECK_ZERO(NvBufSurfaceAllocate(&m_vicInputSurfaces[i], 1, &vicInputSurfaceParams));
#endif
    m_vicInputSurfaces[i]->numFilled = 1;

    CHECK_ZERO(NvBufSurfaceMapEglImage(m_vicInputSurfaces[i], 0));
  }

  // Setup matching rendertarget pool:
  // Create the images in GL, then use EGL to get the backing dmabufs and convert those into NvBufSurfaces.
  // We should then be able to pass those NvBufSurfaces directly to the VIC
  m_rhiSurfaces.clear();
  m_rhiSurfaceEGLImages.clear();
  m_currentSurfaceIndex = 0;

  EGLContext eglThreadCtx = eglGetCurrentContext();
  assert(eglThreadCtx != EGL_NO_CONTEXT);

  for (size_t i = 0; i < kInputBufferCount; ++i) {
    RHISurface::ptr srf = rhi()->newTexture2D(m_width, m_height, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    m_rhiSurfaces.push_back(srf);
#if 0
    EGLAttrib attrs[] = {
      EGL_GL_TEXTURE_LEVEL, 0,
      EGL_NONE
    };

    EGLImage img;
    CHECK_NOT_NULL(img = eglCreateImage(renderBackend->eglDisplay(), eglThreadCtx, EGL_GL_TEXTURE_2D, (EGLClientBuffer) ((/*eliminate size conversion warning*/ intptr_t) static_cast<RHISurfaceGL*>(srf.get())->glId()), attrs));
    m_rhiSurfaceEGLImages.push_back(img);

    surfaceDmaBufInfo info;
    CHECK_TRUE(eglExportDMABUFImageQueryMESA(renderBackend->eglDisplay(), img, &info.fourcc, &info.num_planes, &info.modifiers));

    assert(info.num_planes <= NVBUF_MAX_PLANES);
    CHECK_TRUE(eglExportDMABUFImageMESA(renderBackend->eglDisplay(), img, info.plane_fds, info.plane_strides, info.plane_offsets));

    assert(info.num_planes == 1); // only support single RGB plane
    // We'll populate the NvBufSurface pointer later.

    m_rhiSurfaceDmaBufs.push_back(info);
#endif
  }

  // Create output plane DMABUFs
  {
    NvBufSurfaceAllocateParams encInputSurfaceAllocParams;
    memset(&encInputSurfaceAllocParams, 0, sizeof(encInputSurfaceAllocParams));
    encInputSurfaceAllocParams.params.width = m_width;
    encInputSurfaceAllocParams.params.height = m_height;
    encInputSurfaceAllocParams.params.layout = NVBUF_LAYOUT_PITCH;
    encInputSurfaceAllocParams.params.colorFormat = NVBUF_COLOR_FORMAT_YUV420;
    encInputSurfaceAllocParams.params.memType = NVBUF_MEM_SURFACE_ARRAY;
    encInputSurfaceAllocParams.memtag = NvBufSurfaceTag_VIDEO_ENC;

    m_encOutputPlaneSurfaces.resize(kInputBufferCount);
    for (uint32_t i = 0; i < kInputBufferCount; i++) {
#if L4T_RELEASE < 34
      // use old API
      CHECK_ZERO(NvBufSurfaceCreate(&m_encOutputPlaneSurfaces[i], 1, &encInputSurfaceAllocParams.params));
#else
      CHECK_ZERO(NvBufSurfaceAllocate(&m_encOutputPlaneSurfaces[i], 1, &encInputSurfaceAllocParams));
#endif
      m_encOutputPlaneSurfaces[i]->numFilled = 1;
    }
  }


}

NvEncSession::~NvEncSession() {

  if (m_cudaWorkerThreadRunning) {
    pthread_mutex_lock(&m_gpuSubmissionQueueLock);
    m_gpuSubmissionQueue.push(std::make_pair(-1, EGL_NO_SYNC));
    pthread_cond_broadcast(&m_gpuSubmissionQueueCond);
    pthread_mutex_unlock(&m_gpuSubmissionQueueLock);
    pthread_join(m_cudaWorkerThread, NULL);
    printf("NvEncSession: CUDA worker stopped\n");
    m_cudaWorkerThreadRunning = false;
  }

  // Make sure that the encoder has been completely stopped before deleting anything
  pthread_mutex_lock(&m_stateLock);
  assert(m_startCount == 0);
  pthread_mutex_unlock(&m_stateLock);

  pthread_mutex_destroy(&m_stateLock);
  pthread_mutex_destroy(&m_callbackLock);

  pthread_mutex_destroy(&m_gpuSubmissionQueueLock);
  pthread_cond_destroy(&m_gpuSubmissionQueueCond);

  for (size_t i = 0; i < m_rhiSurfaceDmaBufs.size(); ++i) {
    for (size_t planeIdx = 0; planeIdx < m_rhiSurfaceDmaBufs[i].num_planes; ++i) {
      close(m_rhiSurfaceDmaBufs[i].plane_fds[planeIdx]);
    }
  }
  m_rhiSurfaceDmaBufs.clear();

  for (size_t i = 0; i < m_rhiSurfaceEGLImages.size(); ++i) {
    eglDestroyImage(renderBackend->eglDisplay(), m_rhiSurfaceEGLImages[i]);
  }
  m_rhiSurfaceEGLImages.clear();
  m_rhiSurfaces.clear();

  for (size_t i = 0; i < m_encOutputPlaneSurfaces.size(); ++i) {
    NvBufSurfaceDestroy(m_encOutputPlaneSurfaces[i]);
  }
  m_encOutputPlaneSurfaces.clear();

  for (size_t i = 0; i < m_vicInputSurfaces.size(); ++i) {
    NvBufSurfaceUnMapEglImage(m_vicInputSurfaces[i], 0);
    NvBufSurfaceDestroy(m_vicInputSurfaces[i]);
  }
  m_vicInputSurfaces.clear();
}

size_t NvEncSession::registerEncodedFrameDeliveryCallback(const std::function<void(const char*, size_t, struct timeval&)>& cb) {
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

RHISurface::ptr NvEncSession::acquireSurface() {
  if (pthread_mutex_trylock(&m_stateLock) != 0)
    return RHISurface::ptr();

  RHISurface::ptr res;
  if (!m_rhiSurfaces.empty())
    res =m_rhiSurfaces[m_currentSurfaceIndex];

  pthread_mutex_unlock(&m_stateLock);
  return res;
}

bool NvEncSession::submitSurface(RHISurface::ptr surface, bool blockIfQueueFull) {
  if (pthread_mutex_trylock(&m_stateLock) != 0)
    return false;

  if (m_inShutdown) {
    pthread_mutex_unlock(&m_stateLock);
    return false;
  }

  assert(m_rhiSurfaces[m_currentSurfaceIndex] == surface);

  bool res = true;

  pthread_mutex_lock(&m_gpuSubmissionQueueLock);
  if (m_gpuSubmissionQueue.size() >= m_rhiSurfaces.size()) {
    printf("NvEncSession::submitSurface: queue is full\n");
    res = false; // queue is full
  } else {
    m_gpuSubmissionQueue.push(std::make_pair(m_currentSurfaceIndex, eglCreateSync(renderBackend->eglDisplay(), EGL_SYNC_FENCE, NULL)));
    m_currentSurfaceIndex++;
    if (m_currentSurfaceIndex >= m_rhiSurfaces.size())
      m_currentSurfaceIndex = 0;
  }
  pthread_cond_broadcast(&m_gpuSubmissionQueueCond);
  pthread_mutex_unlock(&m_gpuSubmissionQueueLock);

  pthread_mutex_unlock(&m_stateLock);
  return res;
}

void NvEncSession::cudaWorker() {
  prctl(PR_SET_NAME, "NvEncSessn-CUDA", 0, 0, 0);

  // Setup an EGL share context
  EGLint ctxAttrs[] = {
    EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE
  };

  EGLContext eglCtx = eglCreateContext(renderBackend->eglDisplay(), renderBackend->eglConfig(), renderBackend->eglContext(), ctxAttrs);
  if (!eglCtx) {
    die("rtspServerThreadEntryPoint: unable to create EGL share context\n");
  }

  bool res = eglMakeCurrent(renderBackend->eglDisplay(), EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx);
  if (!res) {
    die("rtspServerThreadEntryPoint: eglMakeCurrent() failed\n");
  }

  // use the default global CUDA context
  cuCtxSetCurrent(cudaContext);

  NvBufSurfTransformConfigParams config_params = {NvBufSurfTransformCompute_VIC, 0, NULL};
  CHECK_ZERO(NvBufSurfTransformSetSessionParams(&config_params));

  // Register EGL images for VIC interop
  std::vector<CUgraphicsResource> vicInputSurfaceResources;
  for (size_t i = 0; i < m_vicInputSurfaces.size(); ++i) {
    CUgraphicsResource pWriteResource = NULL;
    CUDA_CHECK(cuGraphicsEGLRegisterImage(&pWriteResource, m_vicInputSurfaces[i]->surfaceList[0].mappedAddr.eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));
    vicInputSurfaceResources.push_back(pWriteResource);
  }

  while (true) {
    // Wait for next surface index in gpu submission queue
    pthread_mutex_lock(&m_gpuSubmissionQueueLock);
    while (m_gpuSubmissionQueue.empty()) {
      pthread_cond_wait(&m_gpuSubmissionQueueCond, &m_gpuSubmissionQueueLock);
    }
    ssize_t surfaceIdx = m_gpuSubmissionQueue.front().first;
    EGLSyncKHR surfaceSync = m_gpuSubmissionQueue.front().second;
    m_gpuSubmissionQueue.pop();
    pthread_mutex_unlock(&m_gpuSubmissionQueueLock);

    if (surfaceIdx < 0) {
      fprintf(stderr, "NvEncSession::cudaWorker(): thread stop requested\n");
      break;
    }
    RHISurface::ptr surface = m_rhiSurfaces[surfaceIdx];

    NvBuffer* encoderInputBuffer = NULL;

    if (m_encoderOutputPlaneBufferQueue.empty()) {
      struct v4l2_buffer v4l2_buf;
      struct v4l2_plane planes[MAX_PLANES];
      NvBuffer *buffer;
      NvBuffer *shared_buffer;

      memset(&v4l2_buf, 0, sizeof(v4l2_buf));
      memset(planes, 0, sizeof(planes));
      v4l2_buf.m.planes = planes;

      if (m_enc->output_plane.dqBuffer(v4l2_buf, &buffer, &shared_buffer, -1) < 0) {
        die("Failed to dequeue buffer from encoder output plane");
      }

      encoderInputBuffer = m_enc->output_plane.getNthBuffer(v4l2_buf.index);
    } else {
      encoderInputBuffer = m_encoderOutputPlaneBufferQueue.front();
      m_encoderOutputPlaneBufferQueue.pop();
    }

    // Access VIC input surface (will be the target for this copy)
    CUeglFrame eglFrame;
    CUDA_CHECK(cuGraphicsResourceGetMappedEglFrame(&eglFrame, vicInputSurfaceResources[surfaceIdx], 0, 0));

    // Sync with EGL prior to issuing copy
    CUevent hEvent;
    if (CUDA_CHECK_NONFATAL(cuEventCreateFromEGLSync(&hEvent, surfaceSync, CU_EVENT_BLOCKING_SYNC))) { // CU_EVENT_DEFAULT);
      CUDA_CHECK_NONFATAL(cuEventSynchronize(hEvent));
      cuEventDestroy(hEvent);
    }

    eglDestroySync(renderBackend->eglDisplay(), surfaceSync);

    // Map the GL surface for CUDA read access
    CUDA_CHECK(cuGraphicsMapResources(1, &surface->cuGraphicsResource(), 0));

    CUmipmappedArray pReadMip = NULL;
    CUDA_CHECK(cuGraphicsResourceGetMappedMipmappedArray(&pReadMip, surface->cuGraphicsResource()));

    CUarray pReadArray = NULL;
    CUDA_CHECK(cuMipmappedArrayGetLevel(&pReadArray, pReadMip, 0));

    // for debugging
    // CUDA_ARRAY_DESCRIPTOR readArrayDescriptor;
    // CUDA_CHECK(cuArrayGetDescriptor(&readArrayDescriptor, pReadArray));

    CUDA_MEMCPY2D copyDescriptor;
    memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));
    copyDescriptor.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyDescriptor.srcArray = pReadArray;

  #if 0
    // this might work for CU_EGL_FRAME_TYPE_ARRAY? untested.
    copyDescriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyDescriptor.dstArray = eglFrame.frame.pArray[0];
  #else
    // CU_EGL_FRAME_TYPE_PITCH destination
    assert(eglFrame.frameType == CU_EGL_FRAME_TYPE_PITCH);
    copyDescriptor.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyDescriptor.dstDevice = (CUdeviceptr) eglFrame.frame.pPitch[0];
    copyDescriptor.dstPitch = eglFrame.pitch;
  #endif

    copyDescriptor.WidthInBytes = m_vicInputSurfaces[surfaceIdx]->surfaceList[0].planeParams.width[0] * m_vicInputSurfaces[surfaceIdx]->surfaceList[0].planeParams.bytesPerPix[0];
    copyDescriptor.Height = m_vicInputSurfaces[surfaceIdx]->surfaceList[0].height;
    CUDA_CHECK(cuMemcpy2D(&copyDescriptor));

    cuGraphicsUnmapResources(1, &surface->cuGraphicsResource(), 0);

    // Issue transform
    NvBufSurfTransformRect src_rect, dest_rect;
    src_rect.top = 0;
    src_rect.left = 0;
    src_rect.width = m_width;
    src_rect.height = m_height;
    dest_rect.top = 0;
    dest_rect.left = 0;
    dest_rect.width = m_width;
    dest_rect.height = m_height;

    NvBufSurfTransformParams xfParams;
    memset(&xfParams, 0, sizeof(xfParams));

    xfParams.transform_flag = NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_FLIP;
    xfParams.transform_flip = NvBufSurfTransform_FlipY;
    xfParams.transform_filter = NvBufSurfTransformInter_Algo3;
    xfParams.src_rect = &src_rect;
    xfParams.dst_rect = &dest_rect;

    NvBufSurface* encoderInputSrf = m_encOutputPlaneSurfaces[encoderInputBuffer->index];
    NvBufSurfTransform_Error xfErr = NvBufSurfTransform(/*src=*/ m_vicInputSurfaces[surfaceIdx], /*dst=*/ encoderInputSrf, &xfParams);
    if (xfErr != NvBufSurfTransformError_Success) {
      switch (xfErr) {
        case NvBufSurfTransformError_ROI_Error: die("NvBufSurfTransformError_ROI_Error");
        case NvBufSurfTransformError_Invalid_Params: die("NvBufSurfTransformError_Invalid_Params");
        case NvBufSurfTransformError_Execution_Error: die("NvBufSurfTransformError_Execution_Error");
        case NvBufSurfTransformError_Unsupported: die("NvBufSurfTransformError_Unsupported");
        default: die("NvBufSurfTransform bad result %d", xfErr);
      }
    }


    // V4L2 handoff
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, sizeof(planes));

    v4l2_buf.index = encoderInputBuffer->index;
    v4l2_buf.m.planes = planes;
    gettimeofday(&v4l2_buf.timestamp, NULL);

    // bytesused gets reset when the buffer is dequeued, so we have to re-specify it every time before qBuffer
    for (uint32_t planeIdx = 0; planeIdx < encoderInputBuffer->n_planes; ++planeIdx) {
      encoderInputBuffer->planes[planeIdx].fd = m_encOutputPlaneSurfaces[encoderInputBuffer->index]->surfaceList[0].bufferDesc;
      encoderInputBuffer->planes[planeIdx].bytesused = m_encOutputPlaneSurfaces[encoderInputBuffer->index]->surfaceList[0].planeParams.psize[planeIdx];
    }

    int ret = m_enc->output_plane.qBuffer(v4l2_buf, encoderInputBuffer);
    if (ret < 0)
      die("Error while queueing buffer at encoder output plane");
  }

  for (size_t i = 0; i < vicInputSurfaceResources.size(); ++i) {
    cuGraphicsUnregisterResource(vicInputSurfaceResources[i]);
  }

  eglDestroyContext(renderBackend->eglDisplay(), eglCtx);
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

  // It is necessary that Capture Plane format be set before Output Plane format.
  // Set encoder capture plane format. It is necessary to set width and height on the capture plane as well.
  ret = m_enc->setCapturePlaneFormat(m_encoderPixfmt, m_width, m_height, 2 * 1024 * 1024);
  if (ret < 0) die("Could not set output plane format");

  ret = m_enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, m_width, m_height);
  if (ret < 0) die("Could not set output plane format");

  ret = m_enc->setHWPresetType(V4L2_ENC_HW_PRESET_MEDIUM);
  if (ret < 0) die("Could not set encoder hardware quality preset");

  ret = m_enc->setRateControlMode(V4L2_MPEG_VIDEO_BITRATE_MODE_VBR);
  if (ret < 0) die("Could not set rate control mode");

  ret = m_enc->setBitrate(m_bitsPerSecond/2);
  if (ret < 0) die("Could not set bitrate");

  ret = m_enc->setPeakBitrate(m_bitsPerSecond);
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
  m_enc->setIDRInterval(10 /*frames*/);
  m_enc->setMaxPerfMode(1);
  m_enc->setNumBFrames(0); // Disable B-frames for low latency
  // Insert VUI so that the RTSP server can pull framerate information out of it
  //m_enc->setInsertVuiEnabled(true);

  // REQBUF on encoder output plane buffers
  // DMABUF is used here since it is a shared buffer allocated by another component
  // setupPlane can't handle creating the DMABUFs, so we'll do that later.
  ret = m_enc->output_plane.setupPlane(V4L2_MEMORY_DMABUF, kInputBufferCount, false, false);
  if (ret < 0) die("Could not setup encoder output plane");

  // Query, Export and Map the output plane buffers so that we can write
  // encoded data from the buffers
  ret = m_enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, kOutputBufferCount, true, false);
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
  m_enc->capture_plane.setDQThreadCallback(encoder_capture_plane_dq_callback_thunk);
  m_enc->capture_plane.startDQThread(this);

  // Add empty encoder output plane buffers to m_encoderOutputPlaneBufferQueue
  assert(m_enc->output_plane.getNumBuffers() == kInputBufferCount);
  for (uint32_t i = 0; i < m_enc->output_plane.getNumBuffers(); i++) {
    NvBuffer* buf = m_enc->output_plane.getNthBuffer(i);
    m_encoderOutputPlaneBufferQueue.push(buf);
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

  // Start the CUDA worker thread
  while (!m_gpuSubmissionQueue.empty())
    m_gpuSubmissionQueue.pop();

  if (!m_cudaWorkerThreadRunning) {
    pthread_create(&m_cudaWorkerThread, NULL, cudaWorker_thunk, this);
    m_cudaWorkerThreadRunning = true;
  }

  printf("NvEncSession: started.\n");
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

  // Ensure the CUDA worker thread has drained its work queue
  pthread_mutex_lock(&m_gpuSubmissionQueueLock);
  while (!m_gpuSubmissionQueue.empty()) {
    m_gpuSubmissionQueue.pop();
  }
  pthread_mutex_unlock(&m_gpuSubmissionQueueLock);

  // Wait till capture plane DQ Thread finishes
  // i.e. all the capture plane buffers are dequeued
  m_enc->capture_plane.waitForDQThread(2000);

  // Shut down the encoder
  m_enc->abort();

  delete m_enc;
  m_enc = NULL;
  while (!m_encoderOutputPlaneBufferQueue.empty()) m_encoderOutputPlaneBufferQueue.pop();

  printf("NvEncSession: stopped.\n");
  pthread_mutex_unlock(&m_stateLock);
}

bool NvEncSession::encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer) {

  if (m_inShutdown)
    return false; // cancel operations

  if (!v4l2_buf) {
    die("Failed to dequeue buffer from encoder capture plane");
    return false;
  }

  if (buffer->planes[0].bytesused >= 4) {
    // TODO: v4l2_buf->timestamp should have a valid PTS passed through the encoder chain, but using that causes the test player (vlc)
    // to complain about frames being too old to display. might want to offset it somehow?
    // Just calling gettimeofday here to conjure a new PTS seems to work fine

    struct timeval pts;
    gettimeofday(&pts, NULL);

    pthread_mutex_lock(&m_callbackLock);

    const char* data = reinterpret_cast<const char*>(buffer->planes[0].data);
    size_t dataSize = buffer->planes[0].bytesused;

    // Break the delivered buffer up into NAL units, removing start codes, and deliver them individually to downstream clients (RTSP sessions).
    // We only ever get a buffer with an optional SPS (type 7) and PPS (type 8) and then a slice (type 5 or 1)
    const char startCode[] = {0x00, 0x00, 0x00, 0x01};
    const size_t startCodeLength = 4;

    const char* start = data;
    size_t remaining = dataSize;

    if (!memcmp(data, startCode, startCodeLength)) {
      // Skip the first start code
      start += startCodeLength;
      remaining -= startCodeLength;
    } else {
      printf("NvEncSession::encoder_capture_plane_dq_callback: WARNING: delivered buffer is missing start code");
    }

    while (remaining) {
      size_t nextBlockLength = remaining;

      // Split at the next start code in the buffer, if there is one
      const void* nextStartCode = memmem(start, remaining, startCode, startCodeLength);
      if (nextStartCode) {
        nextBlockLength = reinterpret_cast<const char*>(nextStartCode) - start;
      }

      for (const auto& cbIt : m_encodedFrameDeliveryCallbacks) {
        cbIt.second(start, nextBlockLength, pts);
      }

      if (!nextStartCode) // current NALU is the entire rest of the buffer
        break;

      remaining -= nextBlockLength + startCodeLength;
      start += nextBlockLength + startCodeLength;
    }

    pthread_mutex_unlock(&m_callbackLock);
  }

  if (m_enc->capture_plane.qBuffer(*v4l2_buf, NULL) < 0) {
    die("Error while Qing buffer at capture plane");
    return false;
  }

  return true;
}

/*static*/ bool NvEncSession::encoder_capture_plane_dq_callback_thunk(struct v4l2_buffer *v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer, void *arg) {
  return reinterpret_cast<NvEncSession*>(arg)->encoder_capture_plane_dq_callback(v4l2_buf, buffer, shared_buffer);
}

/*static*/ void* NvEncSession::cudaWorker_thunk(void* arg) {
  reinterpret_cast<NvEncSession*>(arg)->cudaWorker();
  return NULL;
}

