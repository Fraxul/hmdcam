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
  //svc->setInputFilename(0, "/mnt/scratch/eyetracking/demo3_200x150.mp4");
  //svc->setInputFilename(0, "/mnt/scratch/eyetracking/ViveProEye_left_200x150.mp4");
  svc->setInputFilename(0, "/mnt/scratch/eyetracking/openEDS_S_1_crop.mp4");

  //cv::VideoWriter videoOut;
  cv::startWindowThread();
  cv::String hWindow = "Eyetracking-Test";
  cv::namedWindow(hWindow);

  //for (size_t frameIdx = 0; frameIdx < 10; ++frameIdx) {
  for (size_t frameIdx = 0; ; ++frameIdx) {
    uint64_t frameStartNs = currentTimeNs();

    if (svc->processFrame()) {
      printf("Frame %zu processing time: %.3fms inference, %.3fms post\n", frameIdx, svc->m_lastFrameProcessingTimeMs, svc->m_lastFramePostProcessingTimeMs);

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

