#include <stdio.h>
#include <signal.h>
#include <cuda.h>
#include "common/SHMSegment.h"
#include "common/DepthMapSHM.h"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <time.h>

static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}


#define CUDA_CHECK(x) checkCUresult(x, #x, __FILE__, __LINE__)
static void checkCUresult(CUresult res, const char* op, const char* file, int line) {
  if (res != CUDA_SUCCESS) {
    const char* errorDesc = NULL;
    cuGetErrorString(res, &errorDesc);
    fprintf(stderr, "%s (%s:%d) returned CUresult %d: %s\n", op, file, line, res, errorDesc);
    abort();
  }
}

static CUdevice s_cudaDevice;
static CUcontext s_cudaContext;

static SHMSegment<DepthMapSHM>* shm;


struct PerViewData {
  cv::cuda::GpuMat m_leftGpu, m_rightGpu;
  cv::cuda::GpuMat m_equalizedLeftGpu, m_equalizedRightGpu;
  cv::cuda::GpuMat m_outputGpu;
  cv::cuda::GpuMat m_disparityFilterTemp;
  cv::cuda::Stream m_stream;
};

int main(int argc, char* argv[]) {
  int ppid = getppid();

  shm = SHMSegment<DepthMapSHM>::openSegment("cuda-dgpu-worker");
  if (!shm) {
    printf("dgpu-worker: Unable to open SHM segment\n");
    return -1;
  }

  CUDA_CHECK(cuInit(0));

  int deviceCount;
  CUDA_CHECK(cuDeviceGetCount(&deviceCount));
  if (!deviceCount) {
    printf("dgpu-worker: No CUDA devices available\n");
    return -1;
  }

  printf("dgpu-worker: %d devices:\n", deviceCount);

  for (int deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
    CUDA_CHECK(cuDeviceGet(&s_cudaDevice, deviceIdx));
    char nameBuf[256];
    memset(nameBuf, 0, 256);
    CUDA_CHECK(cuDeviceGetName(nameBuf, 256, s_cudaDevice));
    printf("dgpu-worker: [%d] %s\n", deviceIdx, nameBuf);
  }

  CUDA_CHECK(cuDevicePrimaryCtxSetFlags(s_cudaDevice, CU_CTX_SCHED_BLOCKING_SYNC)); // maybe CU_CTX_SCHED_SPIN?
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&s_cudaContext, s_cudaDevice));
  CUDA_CHECK(cuCtxSetCurrent(s_cudaContext));

  cv::Ptr<cv::StereoMatcher> m_stereo;
  cv::Ptr<cv::cuda::DisparityBilateralFilter> m_disparityFilter;

  cv::cuda::Event m_processingStartEvent, m_processingEndEvent;


  PerViewData perViewData[DepthMapSHM::maxViews];


  // Signal readyness
  sem_post(&shm->segment()->m_workerReadySem);
  unsigned int lastSettingsGeneration = 0xffffffff;


  while (true) {
    {
      struct timespec ts;
      ts.tv_sec = 1;
      ts.tv_nsec = 0;
      if (sem_timedwait(&shm->segment()->m_workAvailableSem, &ts) < 0) {
        if (errno == ETIMEDOUT) {
          if (kill(ppid, 0) != 0) {
            printf("dgpu-worker: parent process %d has exited\n", ppid);
            return 0;
          }
          continue;
        } else {
          perror("sem_timedwait");
          return -1;
        }
      }
    }

    uint64_t startTime = currentTimeNs();

    if ((!m_stereo) || (shm->segment()->m_settingsGeneration != lastSettingsGeneration)) {
      switch (shm->segment()->m_algorithm) {
        case 0:
          // uses CV_8UC1 disparity
          m_stereo = cv::cuda::createStereoBM(shm->segment()->m_numDisparities, shm->segment()->m_sbmBlockSize);
          break;
        case 1: {
          m_stereo = cv::cuda::createStereoConstantSpaceBP(shm->segment()->m_numDisparities, shm->segment()->m_sbpIterations, shm->segment()->m_sbpLevels, shm->segment()->m_scsbpNrPlane);
        } break;
        case 2:
          m_stereo = cv::cuda::createStereoSGM(0, shm->segment()->m_numDisparities, shm->segment()->m_sgmP1, shm->segment()->m_sgmP2, shm->segment()->m_sgmUniquenessRatio, shm->segment()->m_sgmUseHH4 ? cv::cuda::StereoSGM::MODE_HH4 : cv::cuda::StereoSGM::MODE_HH);
          break;
      };

      if (shm->segment()->m_useDisparityFilter) {
        m_disparityFilter = cv::cuda::createDisparityBilateralFilter(shm->segment()->m_numDisparities, shm->segment()->m_disparityFilterRadius, shm->segment()->m_disparityFilterIterations);
      } else {
        m_disparityFilter.reset();
      }
      lastSettingsGeneration = shm->segment()->m_settingsGeneration;
    }


    // Launch computation for all streams first
    for (size_t viewIdx = 0; viewIdx < shm->segment()->m_activeViewCount; ++viewIdx) {
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];
      PerViewData& viewData = perViewData[viewIdx];

      cv::Mat leftMat(vp.height, vp.width, CV_8UC1, shm->segment()->data() + vp.inputLeftOffset, vp.inputPitchBytes);
      cv::Mat rightMat(vp.height, vp.width, CV_8UC1, shm->segment()->data() + vp.inputRightOffset, vp.inputPitchBytes);

      viewData.m_leftGpu.upload(leftMat, viewData.m_stream);
      viewData.m_rightGpu.upload(rightMat, viewData.m_stream);

      cv::cuda::equalizeHist(viewData.m_leftGpu, viewData.m_equalizedLeftGpu, viewData.m_stream);
      cv::cuda::equalizeHist(viewData.m_rightGpu, viewData.m_equalizedRightGpu, viewData.m_stream);


      // workaround for no common base interface between CUDA stereo remapping algorithms that can handle the CUstream parameter to compute()
      switch (shm->segment()->m_algorithm) {
        case 0:
          static_cast<cv::cuda::StereoBM*>(m_stereo.get())->compute(viewData.m_equalizedLeftGpu, viewData.m_equalizedRightGpu, viewData.m_disparityFilterTemp, viewData.m_stream);
          break;
        case 1:
          static_cast<cv::cuda::StereoConstantSpaceBP*>(m_stereo.get())->compute(viewData.m_equalizedLeftGpu, viewData.m_equalizedRightGpu, viewData.m_disparityFilterTemp, viewData.m_stream);
          break;
        case 2:
          static_cast<cv::cuda::StereoSGM*>(m_stereo.get())->compute(viewData.m_equalizedLeftGpu, viewData.m_equalizedRightGpu, viewData.m_disparityFilterTemp, viewData.m_stream);
          break;
      };

      if (m_disparityFilter) {
        m_disparityFilter->apply(viewData.m_disparityFilterTemp, viewData.m_equalizedLeftGpu, viewData.m_outputGpu, viewData.m_stream);
      } else {
        // Bypass filter, just swap the output pointer
        viewData.m_outputGpu.swap(viewData.m_disparityFilterTemp);
      }
    } // View loop


    // Download all results
    for (size_t viewIdx = 0; viewIdx < shm->segment()->m_activeViewCount; ++viewIdx) {
      PerViewData& viewData = perViewData[viewIdx];
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];

      // StereoBM uses CV_8UC1 disparity map, everything else uses CV_16SC1
      int dispType = (shm->segment()->m_algorithm == 0) ? CV_8UC1 : CV_16SC1;

      cv::Mat dispMat(vp.height, vp.width, dispType, shm->segment()->data() + vp.outputOffset, vp.outputPitchBytes);

      viewData.m_outputGpu.download(dispMat, viewData.m_stream);
    }

    for (size_t viewIdx = 0; viewIdx < shm->segment()->m_activeViewCount; ++viewIdx) {
      PerViewData& viewData = perViewData[viewIdx];
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];

      viewData.m_stream.waitForCompletion();

      shm->flush(vp.outputOffset, vp.outputPitchBytes * vp.height);
    }

    uint64_t deltaT = currentTimeNs() - startTime;
    shm->segment()->m_frameTimeMs = static_cast<double>(deltaT) / 1000000.0;

    // Finished processing all views -- signal completion
    sem_post(&shm->segment()->m_workFinishedSem);
  } // Work loop

  return 0;
}


