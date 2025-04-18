#include "Render.h"
#include "RenderBackend.h"
#include "common/Timing.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/gl/GLCommon.h"

#include <cuda.h>

#include "NvEncSession.h"
#include "liveMedia.hh"
#include "BasicUsageEnvironment.hh"
#include "BufferRingSource.h"
#include "H264VideoNvEncSessionServerMediaSubsession.h"

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

// Streaming server / NvEnc state
std::string rtspURL;
NvEncSession* nvencSession;
TaskScheduler* rtspScheduler;
BasicUsageEnvironment* rtspEnv;
RTSPServer* rtspServer;
ServerMediaSession* rtspMediaSession;
uint64_t rtspRenderIntervalNs = 33333333; // 30fps

void* rtspServerThreadEntryPoint(void* arg) {
  pthread_setname_np(pthread_self(), "RTSP-Server");

  // Initialize EGL share context and CUDA
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

  cuCtxSetCurrent(cudaContext);

  // Set up the RTSP server
  rtspScheduler = BasicTaskScheduler::createNew();
  rtspEnv = BasicUsageEnvironment::createNew(*rtspScheduler);
  OutPacketBuffer::maxSize = 1048576;

  while (true) {
    rtspServer = RTSPServer::createNew(*rtspEnv, 8554);
    if (rtspServer)
      break;

    printf("Failed to create RTSP server: %s. Retrying in 15 seconds...\n", rtspEnv->getResultMsg());
    sleep(15);
  }

  char const* descriptionString = "Live555 embedded stream";
  char const* streamName = "0";
  rtspMediaSession = ServerMediaSession::createNew(*rtspEnv, streamName, streamName, descriptionString);
  rtspMediaSession->addSubsession(H264VideoNvEncSessionServerMediaSubsession::createNew(*rtspEnv, nvencSession));
  rtspServer->addServerMediaSession(rtspMediaSession);

  {
    char* urlTmp = rtspServer->rtspURL(rtspMediaSession);
    printf("RTSP server is listening at %s\n", urlTmp);
    printf("Recommended client configuration for low-latency streaming:\n");
    printf("  ffplay -fflags nobuffer -flags low_delay -framedrop %s\n", urlTmp);

    rtspURL = std::string(urlTmp);
    delete[] urlTmp;
  }

  // Run event loop
  rtspEnv->taskScheduler().doEventLoop();

  // Thread shutdown if the event loop returns (which it shouldn't)
  eglDestroyContext(renderBackend->eglDisplay(), eglCtx);
  return NULL;
}

void RenderInitDebugSurface(uint32_t width, uint32_t height) {
  nvencSession = new NvEncSession(width, height);
  //nvencSession->setBitrate(bitrate);
  //nvencSession->setFramerate(fps_n, fps_d);
  nvencSession->setFramerate(30, 1); // TODO derive this from the screen's framerate.

  // Set up the RTSP server asynchronously.
  pthread_t server_tid;
  pthread_create(&server_tid, NULL, &rtspServerThreadEntryPoint, NULL);
}

bool RenderDebugSubsystemEnabled() {
  return true;
}

RHISurface::ptr renderAcquireDebugSurface() {
  assert(nvencSession != nullptr);

  // RTSP server rendering
  // TODO need to rework the frame handoff so the GPU does the buffer copy
  static uint64_t lastFrameSubmissionTimeNs = 0;
  if (nvencSession->isRunning()) {
    uint64_t now = currentTimeNs();
    if (lastFrameSubmissionTimeNs + rtspRenderIntervalNs <= now) {
      lastFrameSubmissionTimeNs = now;

      return nvencSession->acquireSurface(); // might be NULL anyway if the encoder isn't ready
    }
  }

  return RHISurface::ptr();
}

void renderSubmitDebugSurface(RHISurface::ptr debugSurface) {
  nvencSession->submitSurface(debugSurface);
}

const char* renderDebugURL() {
  return rtspURL.c_str();
}

