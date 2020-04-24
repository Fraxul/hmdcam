#include "NvEncSession.h"
#include "Render.h"
#include "rhi/RHI.h"
#include "rhi/gl/RHISurfaceGL.h"
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaEGL.h>
#include <cudaGL.h>
#include <linux/videodev2.h>
#include <linux/v4l2-controls.h>
#include <stdio.h>
#include <string.h>

#include "nvbuf_utils.h"
#include "NvVideoEncoder.h"
#include "NvVideoConverter.h"
#include "NvUtils.h"
#include "nvgldemo.h"

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

NvEncSession::NvEncSession() : m_width(0), m_height(0), m_bitsPerSecond(8000000),
  m_framerateNumerator(30), m_framerateDenominator(1),
  m_encoderPixfmt(V4L2_PIX_FMT_H264),
  m_inputFormat(kInputFormatNV12),
  m_usingGPUFrameSubmission(false),
  m_encodedFrameDeliveryCallbackIdGen(0),
  m_startCount(0),
  m_inShutdown(false),
  m_inputFrameSize(0),
  m_currentSurfaceIndex(0) {

  pthread_mutex_init(&m_stateLock, NULL);
  pthread_mutex_init(&m_callbackLock, NULL);

  pthread_mutex_init(&m_gpuSubmissionQueueLock, NULL);
  pthread_cond_init(&m_gpuSubmissionQueueCond, NULL);
  pthread_mutex_init(&m_encoderOutputPlaneBufferQueueLock, NULL);
  pthread_cond_init(&m_encoderOutputPlaneBufferQueueCond, NULL);
  pthread_mutex_init(&m_conv0OutputPlaneBufferQueueLock, NULL);
  pthread_cond_init(&m_conv0OutputPlaneBufferQueueCond, NULL);
}

NvEncSession::~NvEncSession() {
  // Make sure that the encoder has been completely stopped before deleting anything
  pthread_mutex_lock(&m_stateLock);
  assert(m_startCount == 0);
  pthread_mutex_unlock(&m_stateLock);

  pthread_mutex_destroy(&m_stateLock);
  pthread_mutex_destroy(&m_callbackLock);

  pthread_mutex_destroy(&m_gpuSubmissionQueueLock);
  pthread_cond_destroy(&m_gpuSubmissionQueueCond);
  pthread_mutex_destroy(&m_encoderOutputPlaneBufferQueueLock);
  pthread_cond_destroy(&m_encoderOutputPlaneBufferQueueCond);
  pthread_mutex_destroy(&m_conv0OutputPlaneBufferQueueLock);
  pthread_cond_destroy(&m_conv0OutputPlaneBufferQueueCond);

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

void NvEncSession::setUseGPUFrameSubmission(bool value) {
  assert(!isRunning());
  m_usingGPUFrameSubmission = value;
}

bool NvEncSession::submitFrame(char* data, size_t length, bool blockIfQueueFull) {
  if (inputFrameSize() == 0)
    return false; // Reject frames submitted before startup is finished

  if (m_inShutdown && length != 0)
    return false; // Reject frames submitted during shutdown

  if (length != 0) {
    assert(!usingGPUFrameSubmission()); // This interface shouldn't be used to submit normal frames if GPU frame submission is enabled
  }


  if (!((length == 0 && m_inShutdown) || length == inputFrameSize())) {
    printf("NvEncSession::submitFrame(): short frame submitted, length %zu (require %zu)\n", length, inputFrameSize());
    return false;
  }

  pthread_mutex_lock(&m_conv0OutputPlaneBufferQueueLock);
  while (m_conv0OutputPlaneBufferQueue.empty()) {
    if (!blockIfQueueFull) {
      pthread_mutex_unlock(&m_conv0OutputPlaneBufferQueueLock);
      return false; // frame dropped
    }

    pthread_cond_wait(&m_conv0OutputPlaneBufferQueueCond, &m_conv0OutputPlaneBufferQueueLock);
  }

  NvBuffer *buffer = m_conv0OutputPlaneBufferQueue.front();
  m_conv0OutputPlaneBufferQueue.pop();
  pthread_mutex_unlock(&m_conv0OutputPlaneBufferQueueLock);

  //printf("NvEncSession::submitFrame() data=%p length=%zu blockIfQueueFull=%u buffer->index=%u\n", data, length, blockIfQueueFull, buffer->index);

  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(planes, 0, sizeof(planes));

  v4l2_buf.index = buffer->index;
  v4l2_buf.m.planes = planes;
  gettimeofday(&v4l2_buf.timestamp, NULL);

  if (data && length) {
    const char* readPtr = data; 
    for (size_t planeIdx = 0; planeIdx < buffer->n_planes; ++planeIdx) {
      NvBuffer::NvBufferPlane &plane = buffer->planes[planeIdx];
      // TODO: this assumes input data is tightly packed.
      size_t bytesPerRow = plane.fmt.bytesperpixel * plane.fmt.width;
      plane.bytesused = plane.fmt.stride * plane.fmt.height;

      char* writePtr = (char *) plane.data;
      for (size_t y = 0; y < plane.fmt.height; ++y) {
        memcpy(writePtr, readPtr, bytesPerRow);
        readPtr += bytesPerRow;
        writePtr += plane.fmt.stride;
      }
    }
  } else {
    for (size_t planeIdx = 0; planeIdx < buffer->n_planes; ++planeIdx) {
      buffer->planes[planeIdx].bytesused = 0;
    }
  }

  int ret = m_conv0->output_plane.qBuffer(v4l2_buf, NULL);
  if (ret < 0)
    die("Error while queueing buffer at output plane");

  return true;
}

RHISurface::ptr NvEncSession::acquireSurface() {
  assert(usingGPUFrameSubmission());
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

  assert(usingGPUFrameSubmission());
  assert(m_rhiSurfaces[m_currentSurfaceIndex] == surface);

  bool res = true;

  pthread_mutex_lock(&m_gpuSubmissionQueueLock);
  if (m_gpuSubmissionQueue.size() >= m_rhiSurfaces.size()) {
    printf("NvEncSession::submitSurface: queue is full\n");
    res = false; // queue is full
  } else {
    m_gpuSubmissionQueue.push(std::make_pair(m_currentSurfaceIndex, eglCreateSync(renderEGLDisplay(), EGL_SYNC_FENCE, NULL)));
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
  // Setup an EGL share context
  EGLContext eglCtx = NvGlDemoCreateShareContext();
  if (!eglCtx) {
    die("rtspServerThreadEntryPoint: unable to create EGL share context\n");
  }

  bool res = eglMakeCurrent(renderEGLDisplay(), EGL_NO_SURFACE, EGL_NO_SURFACE, eglCtx);
  if (!res) {
    die("rtspServerThreadEntryPoint: eglMakeCurrent() failed\n");
  }

  // use the default global CUDA context
  cuCtxSetCurrent(cudaContext);

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
      // Thread stop requested
      break;
    }
    RHISurface::ptr surface = m_rhiSurfaces[surfaceIdx];


    pthread_mutex_lock(&m_conv0OutputPlaneBufferQueueLock);
    while (m_conv0OutputPlaneBufferQueue.empty()) {
      //if (!blockIfQueueFull) {
      //  pthread_mutex_unlock(&m_conv0OutputPlaneBufferQueueLock);
      //  pthread_mutex_unlock(&m_stateLock);
      //  return false; // frame dropped
      //}

      pthread_cond_wait(&m_conv0OutputPlaneBufferQueueCond, &m_conv0OutputPlaneBufferQueueLock);
    }

    NvBuffer *buffer = m_conv0OutputPlaneBufferQueue.front();
    m_conv0OutputPlaneBufferQueue.pop();
    pthread_mutex_unlock(&m_conv0OutputPlaneBufferQueueLock);


    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, sizeof(planes));

    v4l2_buf.index = buffer->index;
    v4l2_buf.m.planes = planes;
    gettimeofday(&v4l2_buf.timestamp, NULL);

    assert(buffer->n_planes == 1); // only support a single RGB plane
    buffer->planes[0].bytesused = buffer->planes[0].fmt.stride * buffer->planes[0].fmt.height;

    // Map the EGLImage for the dmabuf fd for CUDA write access
    EGLImageKHR img = NvEGLImageFromFd(renderEGLDisplay(), buffer->planes[0].fd);

    CUgraphicsResource pWriteResource = NULL;
    CUresult status = cuGraphicsEGLRegisterImage(&pWriteResource, img, CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD);
    if (status != CUDA_SUCCESS)
      die("cuGraphicsEGLRegisterImage failed: %d\n", status);

    CUeglFrame eglFrame;
    status = cuGraphicsResourceGetMappedEglFrame(&eglFrame, pWriteResource, 0, 0);
    if (status != CUDA_SUCCESS) {
      die("cuGraphicsSubResourceGetMappedArray failed: %d\n", status);
    }

    //status = cuCtxSynchronize();
    //if (status != CUDA_SUCCESS) {
    //  die("cuCtxSynchronize failed: %d\n", status);
    //}

    CUevent hEvent;
    status = cuEventCreateFromEGLSync(&hEvent, surfaceSync, CU_EVENT_BLOCKING_SYNC); // CU_EVENT_DEFAULT);
    if (status != CUDA_SUCCESS) {
      die("cuEventCreateFromEGLSync() failed: %d\n", status);
    }

    status = cuEventSynchronize(hEvent);
    if (status != CUDA_SUCCESS) {
      die("cuEventSynchronize() failed: %d\n", status);
    }

    cuEventDestroy(hEvent);
    eglDestroySync(renderEGLDisplay(), surfaceSync);

    // Map the GL surface for CUDA read access
    RHISurfaceGL* glSurface = static_cast<RHISurfaceGL*>(surface.get());

    CUgraphicsResource pReadResource = NULL;
    status = cuGraphicsGLRegisterImage(&pReadResource, glSurface->glId(), glSurface->glTarget(), CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
    if (status != CUDA_SUCCESS) {
      die("cuGraphicsGLRegisterImage() failed: %d\n", status);
    }

    status = cuGraphicsMapResources(1, &pReadResource, 0);
    if (status != CUDA_SUCCESS) {
      die("cuGraphicsMapResources() failed: %d\n", status);
    }

    CUmipmappedArray pReadMip = NULL;
    status = cuGraphicsResourceGetMappedMipmappedArray(&pReadMip, pReadResource);
    if (status != CUDA_SUCCESS) {
      die("cuGraphicsResourceGetMappedMipmappedArray() failed: %d\n", status);
    }

    CUarray pReadArray = NULL;
    status = cuMipmappedArrayGetLevel(&pReadArray, pReadMip, 0);
    if (status != CUDA_SUCCESS) {
      die("cuMipmappedArrayGetLevel() failed: %d\n", status);
    }

    // for debugging
    CUDA_ARRAY_DESCRIPTOR readArrayDescriptor;
    status = cuArrayGetDescriptor(&readArrayDescriptor, pReadArray);
    if (status != CUDA_SUCCESS) {
      die("cuArrayGetDescriptor() failed: %d\n", status);
    }

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

    copyDescriptor.WidthInBytes = buffer->planes[0].fmt.width * buffer->planes[0].fmt.bytesperpixel;
    copyDescriptor.Height = buffer->planes[0].fmt.height;
    status = cuMemcpy2D(&copyDescriptor);
    if (status != CUDA_SUCCESS) {
      die("cuMemcpy2D() failed\n");
    }

    cuGraphicsUnmapResources(1, &pReadResource, 0);
    cuGraphicsUnregisterResource(pReadResource);
    cuGraphicsUnregisterResource(pWriteResource);
    NvDestroyEGLImage(renderEGLDisplay(), img);

    int ret = m_conv0->output_plane.qBuffer(v4l2_buf, NULL);
    if (ret < 0)
      die("Error while queueing buffer at output plane");
  }

  eglDestroyContext(renderEGLDisplay(), eglCtx);
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

  m_conv0 = NvVideoConverter::createVideoConverter("conv0");
  if (!m_conv0) die("Could not create Video Converter");

  // TODO parameterize input format
  uint32_t outputPixfmt = 0;
  switch (m_inputFormat) {
    // V4L2 format enums are in linux/videodev2.h
    case kInputFormatNV12: outputPixfmt = V4L2_PIX_FMT_NV12M; break;
    case kInputFormatRGBX8: outputPixfmt = V4L2_PIX_FMT_ABGR32; break;
    default: die("Invalid input format enum %u", m_inputFormat);
  };

  ret = m_conv0->setOutputPlaneFormat(outputPixfmt, m_width, m_height, V4L2_NV_BUFFER_LAYOUT_PITCH);
  if (ret < 0) die("Could not set output plane format for conv0");

  ret = m_conv0->setCapturePlaneFormat(V4L2_PIX_FMT_YUV420M, m_width, m_height, V4L2_NV_BUFFER_LAYOUT_BLOCKLINEAR);
  if (ret < 0) die("Could not set capture plane format for conv0");

  // XXX: flip the video top to bottom for testing, to prove that the video converter component works
  ret = m_conv0->setFlipMethod(V4L2_FLIP_METHOD_VERT);
  if (ret < 0) die("Could not set flip method");

  // It is necessary that Capture Plane format be set before Output Plane format.
  // Set encoder capture plane format. It is necessary to set width and height on the capture plane as well.
  ret = m_enc->setCapturePlaneFormat(m_encoderPixfmt, m_width, m_height, 2 * 1024 * 1024);
  if (ret < 0) die("Could not set output plane format");

  ret = m_enc->setOutputPlaneFormat(V4L2_PIX_FMT_YUV420M, m_width, m_height);
  if (ret < 0) die("Could not set output plane format");

  ret = m_enc->setBitrate(m_bitsPerSecond);
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
  m_enc->setIDRInterval(60 /*frames*/);
  // Insert VUI so that the RTSP server can pull framerate information out of it
  //m_enc->setInsertVuiEnabled(true);

  if (usingGPUFrameSubmission()) {
    // REQBUF and EXPORT conv0 output plane buffers
    size_t nbufs = 6;
    ret = m_conv0->output_plane.setupPlane(V4L2_MEMORY_MMAP, nbufs, false, false);
    if (ret < 0) die("Error while setting up output plane for conv0");
    // setup matching rendertarget pool
    m_rhiSurfaces.clear();
    m_currentSurfaceIndex = 0;
    for (size_t i = 0; i < nbufs; ++i) {
      RHISurface::ptr srf = rhi()->newTexture2D(m_width, m_height, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
      m_rhiSurfaces.push_back(srf);
    }
  } else {
    // REQBUF, EXPORT and MAP conv0 output plane buffers
    ret = m_conv0->output_plane.setupPlane(V4L2_MEMORY_USERPTR, 6, false, true);
    if (ret < 0) die("Error while setting up output plane for conv0");
  }

  // REQBUF and EXPORT conv0 capture plane buffers
  // No need to MAP since buffer will be shared to next component
  // and not read in application
  ret = m_conv0->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 6, false, false);
  if (ret < 0) die("Error while setting up capture plane for conv0");

  // conv0 output plane STREAMON
  ret = m_conv0->output_plane.setStreamStatus(true);
  if (ret < 0) die("Error in output plane streamon for conv0");

  // conv0 capture plane STREAMON
  ret = m_conv0->capture_plane.setStreamStatus(true);
  if (ret < 0) die("Error in capture plane streamon for conv0");

  // REQBUF on encoder output plane buffers
  // DMABUF is used here since it is a shared buffer allocated by another component
  ret = m_enc->output_plane.setupPlane(V4L2_MEMORY_DMABUF, 6, false, false);
  if (ret < 0) die("Could not setup encoder output plane");

  // Query, Export and Map the output plane buffers so that we can write
  // encoded data from the buffers
  ret = m_enc->capture_plane.setupPlane(V4L2_MEMORY_MMAP, 6, true, false);
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
  m_conv0->capture_plane.setDQThreadCallback(conv0_capture_dqbuf_thread_callback_thunk);
  m_conv0->capture_plane.startDQThread(this);
  m_enc->capture_plane.setDQThreadCallback(encoder_capture_plane_dq_callback_thunk);
  m_enc->capture_plane.startDQThread(this);

  // Enqueue all empty conv0 capture plane buffers
  for (uint32_t i = 0; i < m_conv0->capture_plane.getNumBuffers(); i++) {
    struct v4l2_buffer v4l2_buf;
    struct v4l2_plane planes[MAX_PLANES];

    memset(&v4l2_buf, 0, sizeof(v4l2_buf));
    memset(planes, 0, MAX_PLANES * sizeof(struct v4l2_plane));

    v4l2_buf.index = i;
    v4l2_buf.m.planes = planes;

    ret = m_conv0->capture_plane.qBuffer(v4l2_buf, NULL);
    if (ret < 0) die("Error while queueing buffer at conv0 capture plane");
  }

  // Add all empty encoder output plane buffers to m_encoderOutputPlaneBufferQueue
  for (uint32_t i = 0; i < m_enc->output_plane.getNumBuffers(); i++) {
    m_encoderOutputPlaneBufferQueue.push(m_enc->output_plane.getNthBuffer(i));
  }

  // Add all empty conv0 output plane buffers to m_conv0OutputPlaneBufferQueue
  for (uint32_t i = 0; i < m_conv0->output_plane.getNumBuffers(); i++) {
    m_conv0OutputPlaneBufferQueue.push(m_conv0->output_plane.getNthBuffer(i));
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

  // Compute the input frame size from a conv0 output plane buffer
  {
    NvBuffer* buffer = m_conv0->output_plane.getNthBuffer(0);
    m_inputFrameSize = 0;

    printf("NvEncSession::start(): Input frame planes:\n");
    for (size_t planeIdx = 0; planeIdx < buffer->n_planes; ++planeIdx) {
      NvBuffer::NvBufferPlane &plane = buffer->planes[planeIdx];
      printf("  %zu: %u x %u, %u bytes/pixel, %u bytes\n", planeIdx, plane.fmt.width, plane.fmt.height, plane.fmt.bytesperpixel, plane.fmt.bytesperpixel * plane.fmt.width * plane.fmt.height);

      // TODO: this assumes input data is tightly packed.
      size_t bytesPerRow = plane.fmt.bytesperpixel * plane.fmt.width;
      m_inputFrameSize += bytesPerRow * plane.fmt.height;
    }
    printf("NvEncSession::start(): Input frame %zu bytes total\n", m_inputFrameSize);
  }

  if (usingGPUFrameSubmission()) {
    // Start the CUDA worker thread
    while (!m_gpuSubmissionQueue.empty())
      m_gpuSubmissionQueue.pop();

    pthread_create(&m_cudaWorkerThread, NULL, cudaWorker_thunk, this);
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

  // Submit EOS
  submitFrame(NULL, 0, true);

  if (usingGPUFrameSubmission()) {
    // Stop the CUDA worker thread
    pthread_mutex_lock(&m_gpuSubmissionQueueLock);
    m_gpuSubmissionQueue.push(std::make_pair(-1, EGL_NO_SYNC));
    pthread_cond_broadcast(&m_gpuSubmissionQueueCond);
    pthread_mutex_unlock(&m_gpuSubmissionQueueLock);

    pthread_join(m_cudaWorkerThread, NULL);
    printf("NvEncSession: CUDA worker stopped\n");
  }

  // Wait till capture plane DQ Thread finishes
  // i.e. all the capture plane buffers are dequeued
  m_conv0->waitForIdle(2000);
  m_enc->capture_plane.waitForDQThread(2000);
  delete m_conv0;
  delete m_enc;

  m_conv0 = NULL;
  m_enc = NULL;
  while (!m_encoderOutputPlaneBufferQueue.empty()) m_encoderOutputPlaneBufferQueue.pop();
  while (!m_conv0OutputPlaneBufferQueue.empty()) m_conv0OutputPlaneBufferQueue.pop();

  m_inputFrameSize = 0;
  printf("NvEncSession: stopped.\n");
  pthread_mutex_unlock(&m_stateLock);
}

bool NvEncSession::conv0_output_plane_dq() {
  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];
  NvBuffer *buffer;

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(planes, 0, sizeof(planes));
  v4l2_buf.m.planes = planes;
  v4l2_buf.length = m_enc->output_plane.getNumPlanes();

  if (m_conv0->output_plane.dqBuffer(v4l2_buf, &buffer, NULL, -1) < 0) {
    die("Failed to dequeue buffer from conv0 output plane");
    return false;
  }

  // Add the dequeued buffer to conv0 empty output buffers queue
  pthread_mutex_lock(&m_conv0OutputPlaneBufferQueueLock);

  m_conv0OutputPlaneBufferQueue.push(buffer);
  pthread_cond_broadcast(&m_conv0OutputPlaneBufferQueueCond);
  pthread_mutex_unlock(&m_conv0OutputPlaneBufferQueueLock);

  return true;
}

bool NvEncSession::conv0_capture_dqbuf_thread_callback(struct v4l2_buffer *v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer) {
  NvBuffer *enc_buffer;
  struct v4l2_buffer enc_qbuf;
  struct v4l2_plane planes[MAX_PLANES];

  if (!v4l2_buf) {
    die("Failed to dequeue buffer from conv0 capture plane");
    return false;
  }

  // Get an empty enc output plane buffer from m_encoderOutputPlaneBufferQueue
  pthread_mutex_lock(&m_encoderOutputPlaneBufferQueueLock);
  while (m_encoderOutputPlaneBufferQueue.empty()) {
    pthread_cond_wait(&m_encoderOutputPlaneBufferQueueCond, &m_encoderOutputPlaneBufferQueueLock);
  }

  enc_buffer = m_encoderOutputPlaneBufferQueue.front();
  m_encoderOutputPlaneBufferQueue.pop();
  pthread_mutex_unlock(&m_encoderOutputPlaneBufferQueueLock);

  memset(&enc_qbuf, 0, sizeof(enc_qbuf));
  memset(&planes, 0, sizeof(planes));

  enc_qbuf.index = enc_buffer->index;
  enc_qbuf.m.planes = planes;
  memcpy(&enc_qbuf.timestamp, &v4l2_buf->timestamp, sizeof(struct timeval));

  // A reference to buffer is saved which can be used when
  // buffer is dequeued from enc output plane
  if (m_enc->output_plane.qBuffer(enc_qbuf, buffer) < 0) {
    die("Error queueing buffer on encoder output plane");
    return false;
  }

  if (v4l2_buf->m.planes[0].bytesused == 0) {
    if (m_inShutdown) {
      return false;
    } else {
      printf("conv0_capture_dqbuf_thread_callback: spurious EOS\n");
    }
  }

  // If we're not at EOS, also handle dequeueing buffers from the output plane.
  conv0_output_plane_dq();

  return true;
}

bool NvEncSession::encoder_output_plane_dq() {
  struct v4l2_buffer v4l2_buf;
  struct v4l2_plane planes[MAX_PLANES];
  NvBuffer *buffer;
  NvBuffer *shared_buffer;

  memset(&v4l2_buf, 0, sizeof(v4l2_buf));
  memset(planes, 0, sizeof(planes));
  v4l2_buf.m.planes = planes;
  v4l2_buf.length = m_enc->output_plane.getNumPlanes();

  if (m_enc->output_plane.dqBuffer(v4l2_buf, &buffer, &shared_buffer, -1) < 0) {
    die("Failed to dequeue buffer from encoder output plane");
    return false;
  }

  struct v4l2_buffer conv0_ret_qbuf;

  memset(&conv0_ret_qbuf, 0, sizeof(conv0_ret_qbuf));
  memset(&planes, 0, sizeof(planes));

  // Get the index of the conv0 capture plane shared buffer
  conv0_ret_qbuf.index = shared_buffer->index;
  conv0_ret_qbuf.m.planes = planes;

  // Add the dequeued buffer to encoder empty output buffers queue
  // queue the shared buffer back in conv0 capture plane
  pthread_mutex_lock(&m_encoderOutputPlaneBufferQueueLock);
  if (m_conv0->capture_plane.qBuffer(conv0_ret_qbuf, NULL) < 0) {
    die("Error queueing buffer on conv0 capture plane");
    return false;
  }
  m_encoderOutputPlaneBufferQueue.push(buffer);
  pthread_cond_broadcast(&m_encoderOutputPlaneBufferQueueCond);
  pthread_mutex_unlock(&m_encoderOutputPlaneBufferQueueLock);

  return true;
}

bool NvEncSession::encoder_capture_plane_dq_callback(struct v4l2_buffer *v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer) {

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

  // GOT EOS from encoder. Stop dqthread.
  if (buffer->planes[0].bytesused == 0) {
    if (m_inShutdown)
      return false;
    else
      printf("encoder_capture_plane_dq_callback: spurious EOS\n");
  }

  // Handle dq for output plane here as well. The component seems to lock up (dqBuffer hangs in v4l ioctl)
  // once EOS has been reached, so we only dq on the output plane if we haven't gotten EOS on the capture plane.
  // (The EOS buffer does make it through the component correctly -- only the dq stops working)
  return encoder_output_plane_dq();
}

/*static*/ bool NvEncSession::conv0_capture_dqbuf_thread_callback_thunk(struct v4l2_buffer *v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer, void* arg) {
  return reinterpret_cast<NvEncSession*>(arg)->conv0_capture_dqbuf_thread_callback(v4l2_buf, buffer, shared_buffer);
}

/*static*/ bool NvEncSession::encoder_capture_plane_dq_callback_thunk(struct v4l2_buffer *v4l2_buf, NvBuffer* buffer, NvBuffer* shared_buffer, void *arg) {
  return reinterpret_cast<NvEncSession*>(arg)->encoder_capture_plane_dq_callback(v4l2_buf, buffer, shared_buffer);
}

/*static*/ void* NvEncSession::cudaWorker_thunk(void* arg) {
  reinterpret_cast<NvEncSession*>(arg)->cudaWorker();
  return NULL;
}

