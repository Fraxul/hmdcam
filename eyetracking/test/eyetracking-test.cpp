#include "EyeTrackingService.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <unistd.h>
#include <signal.h>

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

#include "common/Timing.h"

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)
#include <cuda.h>
#define CUDA_CHECK(x) checkCUresult(x, #x, __FILE__, __LINE__, true)
#define CUDA_CHECK_NONFATAL(x) checkCUresult(x, #x, __FILE__, __LINE__, false)
#define CUDA_SAFE_FREE(x) do { if (x) { cuMemFree(x); x = 0; } } while(0)
#define NPP_CHECK(x) do { NppStatus _nppStatus = x; if (_nppStatus < 0) { fprintf(stderr, "%s (%s:%d) returned NppStatus %d\n", #x, __FILE__, __LINE__, _nppStatus); abort(); } } while(0)
bool checkCUresult(CUresult res, const char* op, const char* file, int line, bool fatal) {
  if (res != CUDA_SUCCESS) {
    const char* errorDesc = NULL;
    cuGetErrorString(res, &errorDesc);
    fprintf(stderr, "%s (%s:%d) returned CUresult %d: %s\n", op, file, line, res, errorDesc);
    if (fatal)
      abort();
    return false;
  }
  return true;
}


bool want_quit = false;
static void signal_handler(int) {
  want_quit = true;

  // Restore signal handlers so the program is still interruptable if clean shutdown gets stuck
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);
}

// CUDA
CUdevice cudaDevice;
CUcontext cudaContext;

int main(int argc, char* argv[]) {
  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);


  // CUDA init
  {
    cuInit(0);

    cuDeviceGet(&cudaDevice, 0);
    char devName[512];
    cuDeviceGetName(devName, 511, cudaDevice);
    devName[511] = '\0';
    printf("CUDA device: %s\n", devName);

    cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice);
    cuCtxSetCurrent(cudaContext);
  }

  EyeTrackingService* svc = new EyeTrackingService();
  svc->m_debugShowFeedbackView = true; // Required to populate getDebugViewForEye

  //svc->setInputFilename(0, "/mnt/scratch/eyetracking/demo3_200x150.mp4");
  //svc->setInputFilename(0, "/mnt/scratch/eyetracking/ViveProEye_left_200x150.mp4");
  if (argc > 1) {
    printf("Using input filename %s\n", argv[1]);
    svc->setInputFilename(0, argv[1]);
  } else {
    svc->setInputFilename(0, "/mnt/scratch/eyetracking/openEDS_S_1.mp4");
  }

  //cv::VideoWriter videoOut;
  cv::startWindowThread();
  cv::String hWindow = "Eyetracking-Test";
  cv::namedWindow(hWindow);

#if 0
  int mm2px_int = static_cast<int>(svc->m_processingState[0].m_mm2px_scaling);
  auto mm2px_callback = [](int newValue, void* svc_) {
    EyeTrackingService* svc = reinterpret_cast<EyeTrackingService*>(svc_);
    svc->m_processingState[0].m_mm2px_scaling = newValue;

    svc->m_processingState[0].m_eyeModelFitter.focal_length =
      svc->m_processingState[0].m_focalLength * svc->m_processingState[0].m_mm2px_scaling;
  };

  cv::createTrackbar("mm2px", hWindow, &mm2px_int, 500, mm2px_callback, svc);
#endif

  //cv::createTrackbar("Circular crop X", hWindow, &svc->m_processingState[0].m_cropCenter.x, 639, nullptr, nullptr);
  //cv::createTrackbar("Circular crop Y", hWindow, &svc->m_processingState[0].m_cropCenter.y, 479, nullptr, nullptr);
  //cv::createTrackbar("Circular crop Radius", hWindow, (int*) &svc->m_processingState[0].m_cropRadius, 240, nullptr, nullptr);

  //for (size_t frameIdx = 0; frameIdx < 10; ++frameIdx) {
  for (size_t frameIdx = 0; ; ++frameIdx) {
    uint64_t frameStartNs = currentTimeNs();

    if (svc->processFrame()) {
      if ((frameIdx & 127) == 0) {
        printf("Frame %zu: %s\n", frameIdx, svc->getDebugPerfStatsForEye(0));
      }

      cv::Mat& dbgView = svc->getDebugViewForEye(0);
      if (!dbgView.empty()) {
/*
        if (!videoOut.isOpened()) {
          videoOut.open("eyetracking-test-out.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), 30, dbgView.size());
        }
        videoOut.write(dbgView);
*/
        cv::imshow(hWindow, dbgView);
      }
    }

    uint64_t frameEndNs = currentTimeNs();
    uint64_t frameTimeNs = (frameEndNs - frameStartNs);
    //const uint64_t frameTargetTimeNs = 11'111'111; // 11.1ms, 90fps
    const uint64_t frameTargetTimeNs = (1'000'000'000 / 30); // 30fps
    int delayMs = std::max<int>((frameTimeNs < frameTargetTimeNs) ? ((frameTargetTimeNs - frameTimeNs) / 1'000'000U) : 1, 1);
    int key = cv::waitKey(delayMs);
    switch (key) {
      case 'q':
        want_quit = true;
        break;

      case 'r':
        printf("Attempting refine\n");
        svc->m_processingState[0].m_eyeModelFitter.refine_with_inliers();
        break;

      case 'c':
        printf("Clearing samples and model\n");
        svc->m_processingState[0].m_eyeModelFitter.reset();
        svc->m_processingState[0].m_eyeFitterSamples.clear();
        break;

      default:
        printf("Key = %c %d 0x%x\n", key, key, key);
        break;

      case -1:
        // No key pressed
        break;
    }

    //if (frameTimeNs < frameTargetTimeNs) {
    //  delayNs(frameTargetTimeNs - frameTimeNs);
    //}

    if (want_quit) {
      //videoOut.release();
      break;
    }
  }

  delete svc;

  return 0;
}

