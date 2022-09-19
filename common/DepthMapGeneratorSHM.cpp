#ifdef HAVE_OPENCV_CUDA
#include "common/DepthMapGeneratorSHM.h"
#include "imgui.h"
#include "implot/implot.h"
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/Timing.h"
#include "common/glmCvInterop.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/RHICVInterop.h"
#include "rhi/gl/GLCommon.h"
#include <opencv2/cvconfig.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda.h>
#include <npp.h>
#include <epoxy/gl.h> // epoxy_is_desktop_gl

#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <limits.h>
#include <string>

#define NUM_DISP 128 //Max disparity. Must be in {64, 128, 256} for CUDA algorithms.

DepthMapGeneratorSHM::DepthMapGeneratorSHM(DepthMapGeneratorBackend _backend) : DepthMapGenerator(_backend) {

  // running at quarter res, approx
  m_algoDownsampleX = 4;
  m_algoDownsampleY = 4;
  m_maxDisparity = NUM_DISP;

  m_depthMapSHM = SHMSegment<DepthMapSHM>::createSegment("depth-worker", 16*1024*1024);

  printf("Waiting for depth worker...\n");

  if (getenv("DEPTH_WORKER_ATTACH")) {
    // Debug support for launching the worker process externally (through a debugger or NSight)
    printf("Waiting for externally-spawned depth worker...\n");
    int res = sem_wait(&m_depthMapSHM->segment()->m_workerReadySem);
    if (res != 0) {
      perror("sem_wait()");
      abort();
    }
  } else {
    int timeout_sec = 5;
    {
      const char* timeoutEnvStr = getenv("DEPTH_WORKER_TIMEOUT");
      if (timeoutEnvStr) {
        int timeoutEnv = atoi(timeoutEnvStr);
        if (timeoutEnv <= 0) {
          timeout_sec = UINT_MAX;
          fprintf(stderr, "DepthMapGeneratorSHM: DEPTH_WORKER_TIMEOUT <= 0, waiting forever\n");
        } else {
          timeout_sec = timeoutEnv;
          fprintf(stderr, "DepthMapGeneratorSHM: DEPTH_WORKER_TIMEOUT is %u seconds\n", timeout_sec);
        }
      }
    }

    int pid = spawnDepthWorker();
    waitForDepthWorkerReady(pid, &m_depthMapSHM->segment()->m_workerReadySem, timeout_sec);
  }
}

int DepthMapGeneratorSHM::spawnDepthWorker() {
  char* exepath = realpath("/proc/self/exe", NULL);
  assert(exepath);

  std::string workerBin = std::string(dirname(exepath));
  free(exepath);

  switch (m_backend) {
    case kDepthBackendNone:
      assert(false && "spawnDepthWorker: can't spawn process for backend type kDepthBackendNone");
      break;

    case kDepthBackendDGPU:
      workerBin += "/dgpu-worker";
      break;

    case kDepthBackendDepthAI:
      workerBin += "/depthai-worker";
      break;

    default:
      assert(false && "spawnDepthWorker: invalid backend enum");
  };

  int pid = vfork();
  if (!pid) {
    // spawn child process
    char* argv0 = const_cast<char*>(workerBin.c_str());
    char* args[] = { const_cast<char*>(argv0), NULL };
    if (-1 == execv(argv0, args)) {
      printf("execv() failed: %s\n", strerror(errno));
      _exit(-1);
    }
  }

  printf("Spawning Depth worker binary %s as PID %d\n", workerBin.c_str(), pid);
  return pid;
}

void DepthMapGeneratorSHM::waitForDepthWorkerReady(int pid, sem_t* sem, unsigned int timeout_sec) {
  char err[128];

  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  for (unsigned int i = 0; i < timeout_sec; ++i) {
    ts.tv_sec += 1;
    int res = sem_timedwait(sem, &ts);
    if (res == 0)
      return; // OK
    else if (errno != ETIMEDOUT) {
      sprintf(err, "waitForDepthWorkerReady(): sem_timedwait: %s", strerror(errno));
      throw std::runtime_error(err);
    }
    int wstatus;
    if (waitpid(pid, &wstatus, WNOHANG) > 0) {
      if (WIFEXITED(wstatus)) {
        sprintf(err, "Depth worker exited; status %d", WEXITSTATUS(wstatus));
      } else if (WIFSIGNALED(wstatus)) {
        sprintf(err, "Depth worker exited; signal %d", WTERMSIG(wstatus));
      } else {
        sprintf(err, "Depth worker exited; unknown reason");
      }
      throw std::runtime_error(err);
    }

  }
  throw std::runtime_error("Timed out waiting for Depth worker to initialize");
}

DepthMapGeneratorSHM::~DepthMapGeneratorSHM() {
  delete m_depthMapSHM;
}

#define readNode(node, settingName) cv::read(node[#settingName], m_depthMapSHM->segment()->m_##settingName, m_depthMapSHM->segment()->m_##settingName)
void DepthMapGeneratorSHM::internalLoadSettings(cv::FileStorage& fs) {
  cv::FileNode dgpu = fs["dgpu"];
  if (dgpu.isMap()) {
    readNode(dgpu, algorithm);
    readNode(dgpu, useDisparityFilter);
    readNode(dgpu, disparityFilterRadius);
    readNode(dgpu, disparityFilterIterations);
    readNode(dgpu, sbmBlockSize);
    readNode(dgpu, sbpIterations);
    readNode(dgpu, sbpLevels);
    readNode(dgpu, scsbpNrPlane);
    readNode(dgpu, sgmP1);
    readNode(dgpu, sgmP2);
    readNode(dgpu, sgmUniquenessRatio);
    readNode(dgpu, sgmUseHH4);
  }

  cv::FileNode dai = fs["depthai"];
  if (dai.isMap()) {
    readNode(dai, confidenceThreshold);
    readNode(dai, medianFilter);
    readNode(dai, bilateralFilterSigma);
    readNode(dai, enableLRCheck);
    readNode(dai, leftRightCheckThreshold);
  }
}
#undef readNode

#define writeNode(fileStorage, settingName) fileStorage.write(#settingName, m_depthMapSHM->segment()->m_##settingName)
void DepthMapGeneratorSHM::internalSaveSettings(cv::FileStorage& fs) {
  fs.startWriteStruct(cv::String("dgpu"), cv::FileNode::MAP, cv::String());
    writeNode(fs, algorithm);
    writeNode(fs, useDisparityFilter);
    writeNode(fs, disparityFilterRadius);
    writeNode(fs, disparityFilterIterations);
    writeNode(fs, sbmBlockSize);
    writeNode(fs, sbpIterations);
    writeNode(fs, sbpLevels);
    writeNode(fs, scsbpNrPlane);
    writeNode(fs, sgmP1);
    writeNode(fs, sgmP2);
    writeNode(fs, sgmUniquenessRatio);
    writeNode(fs, sgmUseHH4);
  fs.endWriteStruct();

  fs.startWriteStruct(cv::String("depthai"), cv::FileNode::MAP, cv::String());
    writeNode(fs, confidenceThreshold);
    writeNode(fs, medianFilter);
    writeNode(fs, bilateralFilterSigma);
    writeNode(fs, enableLRCheck);
    writeNode(fs, leftRightCheckThreshold);
  fs.endWriteStruct();
}
#undef writeNode

void DepthMapGeneratorSHM::internalUpdateViewData() {
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
      CameraSystem::View& v = m_cameraSystem->viewAtIndex(viewIdx);
      auto vd = viewDataAtIndex(viewIdx);

      CameraSystem::Camera& cL = m_cameraSystem->cameraAtIndex(v.cameraIndices[0]);
      CameraSystem::Camera& cR = m_cameraSystem->cameraAtIndex(v.cameraIndices[1]);

      // TODO validate CameraSystem:::updateViewStereoDistortionParameters against the distortion map initialization code here
      cv::Size imageSize = cv::Size(inputWidth(), inputHeight());

      {
        cv::Mat m1, m2;
        cv::initUndistortRectifyMap(cL.intrinsicMatrix, cL.distCoeffs, v.stereoRectification[0], v.stereoProjection[0], imageSize, CV_32F, m1, m2);
        vd->m_leftMap1_gpu.upload(m1); vd->m_leftMap2_gpu.upload(m2);
        cv::initUndistortRectifyMap(cR.intrinsicMatrix, cR.distCoeffs, v.stereoRectification[1], v.stereoProjection[1], imageSize, CV_32F, m1, m2);
        vd->m_rightMap1_gpu.upload(m1); vd->m_rightMap2_gpu.upload(m2);

      }
      // vd->m_disparityTexture = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R16i));

      //Set up what matrices we can to prevent dynamic memory allocation.
      vd->rectLeft_gpu = cv::cuda::GpuMat(inputHeight(), inputWidth(), CV_8U);
      vd->rectRight_gpu = cv::cuda::GpuMat(inputHeight(), inputWidth(), CV_8U);

      vd->resizedLeft_gpu = cv::cuda::GpuMat(internalHeight(), internalWidth(), CV_8U);
      vd->resizedRight_gpu = cv::cuda::GpuMat(internalHeight(), internalWidth(), CV_8U);

      if (vd->m_isVerticalStereo) {
        vd->resizedTransposedLeft_gpu = cv::cuda::GpuMat(internalWidth(), internalHeight(), CV_8U);
        vd->resizedTransposedRight_gpu = cv::cuda::GpuMat(internalWidth(), internalHeight(), CV_8U);
      }

  }
}


void DepthMapGeneratorSHM::internalProcessFrame() {
  if (m_enableProfiling) {
    if (m_haveValidProfilingData) {
      cuEventElapsedTime(&m_profilingData.m_setupTimeMs, cv::cuda::EventAccessor::getEvent(m_setupStartEvent), cv::cuda::EventAccessor::getEvent(m_setupFinishedEvent));
      if (m_profilingData.m_processingTimedOutThisFrame == 0) {
        cuEventElapsedTime(&m_profilingData.m_copyTimeMs, cv::cuda::EventAccessor::getEvent(m_copyStartEvent), cv::cuda::EventAccessor::getEvent(m_processingFinishedEvent));
      }
    }
    m_profilingDataBuffer.push_back(m_profilingData);
  } else {
    m_haveValidProfilingData = false;
  }

  m_profilingData = {}; // clear for next frame

  if (m_didChangeSettings) {
    if (m_backend == kDepthBackendDGPU) {
      switch (m_depthMapSHM->segment()->m_algorithm) {
        default:
          m_depthMapSHM->segment()->m_algorithm = 0;
        case 0:
          // uses CV_8UC1 disparity
          m_disparityPrescale = 1.0f; // / 16.0f;
          m_disparityBytesPerPixel = 1;
          break;
        case 1: {
          m_disparityPrescale = 1.0f; // / 16.0f;
          m_disparityBytesPerPixel = 2;
        } break;
        case 2:
          m_disparityPrescale = 1.0f / 16.0f; // TODO: not sure if this is correct -- matches results from CSBP, roughly.
          m_disparityBytesPerPixel = 2;
          break;
      };
    } else if (m_backend == kDepthBackendDepthAI) {
      m_disparityPrescale = 1.0f;
      m_disparityBytesPerPixel = 1;
    } else {
      assert(false && "DepthMapGenerator::processFrame(): settings update not implemented for this depth backend");
    }

    for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
      auto vd = viewDataAtIndex(viewIdx);
      if (!vd->m_isStereoView)
        continue;

      vd->updateDisparityTexture(internalWidth(), internalHeight(), m_disparityBytesPerPixel == 1 ? kSurfaceFormat_R8i : kSurfaceFormat_R16i);
    }

    // Increment generation so the worker re-reads the settings when starting the next frame
    m_depthMapSHM->segment()->m_settingsGeneration += 1;

    m_didChangeSettings = false;
  }

  // Setup: Copy input camera surfaces to CV GpuMats, resize/grayscale/normalize
  if (m_enableProfiling) {
    m_setupStartEvent.record(m_globalStream);
  }

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);

    if (!vd->m_isStereoView)
      continue;

    cv::cuda::GpuMat origLeft_gpu = m_cameraSystem->cameraProvider()->gpuMatGreyscale(m_cameraSystem->viewAtIndex(viewIdx).cameraIndices[0]);
    cv::cuda::GpuMat origRight_gpu = m_cameraSystem->cameraProvider()->gpuMatGreyscale(m_cameraSystem->viewAtIndex(viewIdx).cameraIndices[1]);

    cv::cuda::remap(origLeft_gpu, vd->rectLeft_gpu, vd->m_leftMap1_gpu, vd->m_leftMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue=*/ cv::Scalar(), m_globalStream);
    cv::cuda::remap(origRight_gpu, vd->rectRight_gpu, vd->m_rightMap1_gpu, vd->m_rightMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue=*/ cv::Scalar(), m_globalStream);

    cv::cuda::resize(vd->rectLeft_gpu, vd->resizedLeft_gpu, cv::Size(internalWidth(), internalHeight()), 0, 0, cv::INTER_LINEAR, m_globalStream);
    cv::cuda::resize(vd->rectRight_gpu, vd->resizedRight_gpu, cv::Size(internalWidth(), internalHeight()), 0, 0, cv::INTER_LINEAR, m_globalStream);

    if (vd->m_isVerticalStereo) {
      // cv::cuda::transpose is unusable due to forced CPU-GPU sync when switching the CUDA stream that NPPI is targeting, so we skip the CV wrappers and use NPPI directly.
      NppiSize sz;
      sz.width  = vd->resizedLeft_gpu.cols;
      sz.height = vd->resizedLeft_gpu.rows;

      if (nppGetStream() != ((cudaStream_t) m_globalStream.cudaPtr())) {
        nppSetStream((cudaStream_t) m_globalStream.cudaPtr());
      }

      nppiTranspose_8u_C1R(vd->resizedLeft_gpu.ptr<Npp8u>(), static_cast<int>(vd->resizedLeft_gpu.step), vd->resizedTransposedLeft_gpu.ptr<Npp8u>(), static_cast<int>(vd->resizedTransposedLeft_gpu.step), sz);
      nppiTranspose_8u_C1R(vd->resizedRight_gpu.ptr<Npp8u>(), static_cast<int>(vd->resizedRight_gpu.step), vd->resizedTransposedRight_gpu.ptr<Npp8u>(), static_cast<int>(vd->resizedTransposedRight_gpu.step), sz);
    }
  }

  if (m_enableProfiling) {
    m_setupFinishedEvent.record(m_globalStream);
  }


  // Wait for previous processing to finish
  {
    struct timespec timeout;
    clock_gettime(CLOCK_REALTIME, &timeout);
    timeout.tv_nsec += 1000000ULL; // 1ms wait time max

    if (timeout.tv_nsec >= 1000000000ULL) {
      // handle carry
      timeout.tv_sec += 1;
      timeout.tv_nsec -= 1000000000ULL;
    }
    uint64_t wait_start = currentTimeNs();
    int res = sem_timedwait(&m_depthMapSHM->segment()->m_workFinishedSem, &timeout);
    m_profilingData.m_syncTimeMs = deltaTimeMs(wait_start, currentTimeNs());
    if (res < 0) {
      m_profilingData.m_processingTimedOutThisFrame = 1.0f;

      return;
    }

    m_profilingData.m_processingTimedOutThisFrame = 0.0f;
  }

  m_profilingData.m_algoTimeMs = m_depthMapSHM->segment()->m_frameTimeMs;

  // Setup view data in the SHM segment
  m_depthMapSHM->segment()->m_activeViewCount = 0;
  size_t lastOffset = 0;

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    vd->m_shmViewIndex = m_depthMapSHM->segment()->m_activeViewCount;
    m_depthMapSHM->segment()->m_activeViewCount += 1;

    DepthMapSHM::ViewParams& viewParams = m_depthMapSHM->segment()->m_viewParams[vd->m_shmViewIndex];
    cv::cuda::GpuMat& leftGpu = vd->m_isVerticalStereo ? vd->resizedTransposedLeft_gpu : vd->resizedLeft_gpu;
    cv::cuda::GpuMat& rightGpu = vd->m_isVerticalStereo ? vd->resizedTransposedRight_gpu : vd->resizedRight_gpu;

    viewParams.width = leftGpu.cols;
    viewParams.height = leftGpu.rows;
    viewParams.inputPitchBytes = leftGpu.step;
    viewParams.outputPitchBytes = viewParams.width * m_disparityBytesPerPixel; // Tightly packed output so it can be passed to RHI::loadTextureData

    // Allocate space in the SHM region for the I/O buffers.
    size_t inputBufferSize = ((viewParams.height * viewParams.inputPitchBytes) + 4095) & (~4095); // rounded up to pagesize
    size_t outputBufferSize = ((viewParams.height * viewParams.outputPitchBytes) + 4095) & (~4095);

    viewParams.inputOffset[0]  = lastOffset;
    viewParams.inputOffset[1] = viewParams.inputOffset[0] + inputBufferSize;
    viewParams.outputOffset = viewParams.inputOffset[1] + inputBufferSize;

    lastOffset = viewParams.outputOffset + outputBufferSize;

    cv::Mat leftMat(viewParams.height, viewParams.width, CV_8UC1, m_depthMapSHM->segment()->data() + viewParams.inputOffset[0], viewParams.inputPitchBytes);
    cv::Mat rightMat(viewParams.height, viewParams.width, CV_8UC1, m_depthMapSHM->segment()->data() + viewParams.inputOffset[1], viewParams.inputPitchBytes);
    cv::Mat dispMat(viewParams.height, viewParams.width, (m_disparityBytesPerPixel == 1) ? CV_8UC1 : CV_16UC1, m_depthMapSHM->segment()->data() + viewParams.outputOffset, viewParams.outputPitchBytes);

    leftGpu.download(leftMat, m_globalStream);
    rightGpu.download(rightMat, m_globalStream);
  }

  // Write current frame to the SHM segment
  m_globalStream.waitForCompletion(); // finish CUDA->SHM copies

  // Flush all modified regions
  m_depthMapSHM->flush(0, m_depthMapSHM->segment()->m_dataOffset); // header
  for (size_t shmViewIdx = 0; shmViewIdx < m_depthMapSHM->segment()->m_activeViewCount; ++shmViewIdx) {
    DepthMapSHM::ViewParams& viewParams = m_depthMapSHM->segment()->m_viewParams[shmViewIdx];

    m_depthMapSHM->flush(viewParams.inputOffset[0], viewParams.inputPitchBytes * viewParams.width);
    m_depthMapSHM->flush(viewParams.inputOffset[1], viewParams.inputPitchBytes * viewParams.width);
  }


  // Signal worker to start depth processing
  sem_post(&m_depthMapSHM->segment()->m_workAvailableSem);

  if (m_enableProfiling) {
    m_copyStartEvent.record(m_globalStream);
  }

  // Copy results to render surfaces
  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    DepthMapSHM::ViewParams& viewParams = m_depthMapSHM->segment()->m_viewParams[vd->m_shmViewIndex];

    if (m_debugDisparityCPUAccessEnabled)
      vd->ensureDebugCPUAccessEnabled(m_disparityBytesPerPixel);

    if (vd->m_isVerticalStereo) {
      cv::Mat transposedDispMat(viewParams.height, viewParams.width, (m_disparityBytesPerPixel == 1) ? CV_8UC1 : CV_16UC1, m_depthMapSHM->segment()->data() + viewParams.outputOffset, viewParams.outputPitchBytes);
      cv::Mat dispMat = transposedDispMat.t(); // TODO might be more efficient to transpose in the DGPUWorker while the data is still on the GPU
      rhi()->loadTextureData(vd->m_disparityTexture, (m_disparityBytesPerPixel == 1) ? kVertexElementTypeByte1 : kVertexElementTypeShort1, dispMat.data);

      if (m_debugDisparityCPUAccessEnabled)
        memcpy(vd->m_debugCPUDisparity, dispMat.data, m_disparityBytesPerPixel * viewParams.width * viewParams.height);

    } else {
      rhi()->loadTextureData(vd->m_disparityTexture, (m_disparityBytesPerPixel == 1) ? kVertexElementTypeByte1 : kVertexElementTypeShort1, m_depthMapSHM->segment()->data() + viewParams.outputOffset);

      if (m_debugDisparityCPUAccessEnabled)
        memcpy(vd->m_debugCPUDisparity, m_depthMapSHM->segment()->data() + viewParams.outputOffset, m_disparityBytesPerPixel * viewParams.width * viewParams.height);
    }

    if (m_populateDebugTextures) {
      if (!vd->m_leftGray)
        vd->m_leftGray = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      if (!vd->m_rightGray)
        vd->m_rightGray = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      RHICUDA::copyGpuMatToSurface(vd->resizedLeft_gpu, vd->m_leftGray, m_globalStream);
      RHICUDA::copyGpuMatToSurface(vd->resizedRight_gpu, vd->m_rightGray, m_globalStream);
    }
  }

  internalGenerateDisparityMips();

  if (m_enableProfiling) {
    m_processingFinishedEvent.record(m_globalStream);
    m_haveValidProfilingData = true;
  }

#ifndef L4T_RELEASE_MAJOR
  // stupid workaround for profiling on desktop RDMAclient
  m_globalStream.waitForCompletion();
#endif
}

void DepthMapGeneratorSHM::internalRenderIMGUI() {
  DepthMapSHM* shm = m_depthMapSHM->segment();
  if (m_backend == kDepthBackendDGPU) {
    ImGui::Text("Stereo Algorithm");
    m_didChangeSettings |= ImGui::RadioButton("BM", &shm->m_algorithm, 0);
    m_didChangeSettings |= ImGui::RadioButton("ConstantSpaceBeliefPropagation", &shm->m_algorithm, 1);
    m_didChangeSettings |= ImGui::RadioButton("SGM", &shm->m_algorithm, 2);

    switch (shm->m_algorithm) {
      case 0: // StereoBM
        if ((m_didChangeSettings |= ImGui::InputInt("Block Size (odd)", &shm->m_sbmBlockSize, /*step=*/2))) {
          shm->m_sbmBlockSize |= 1; // enforce odd blockSize
        }
        break;
      case 1: // StereoConstantSpaceBP
        m_didChangeSettings |= ImGui::SliderInt("nr_plane", &shm->m_scsbpNrPlane, 1, 16);
        m_didChangeSettings |= ImGui::SliderInt("SBP Iterations", &shm->m_sbpIterations, 1, 8);
        if ((m_didChangeSettings |= ImGui::SliderInt("SBP Levels", &shm->m_sbpLevels, 1, 8))) {
          if (shm->m_sbpLevels > 4) // more than 4 levels seems to trigger an internal crash
            shm->m_sbpLevels = 4;
        }
        break;
      case 2: // StereoSGM
        m_didChangeSettings |= ImGui::SliderInt("SGM P1", &shm->m_sgmP1, 1, 255);
        m_didChangeSettings |= ImGui::SliderInt("SGM P2", &shm->m_sgmP2, 1, 255);
        m_didChangeSettings |= ImGui::SliderInt("SGM Uniqueness Ratio", &shm->m_sgmUniquenessRatio, 5, 15);
        break;
    };

    m_didChangeSettings |= ImGui::Checkbox("Disparity filter (GPU)", &shm->m_useDisparityFilter);

    if (shm->m_useDisparityFilter) {
      if ((m_didChangeSettings |= ImGui::SliderInt("Filter Radius (odd)", &shm->m_disparityFilterRadius, 1, 9))) {
        shm->m_disparityFilterRadius |= 1; // enforce odd filter size
      }
      m_didChangeSettings |= ImGui::SliderInt("Filter Iterations", &shm->m_disparityFilterIterations, 1, 8);
    }
  } else if (m_backend == kDepthBackendDepthAI) {
    m_didChangeSettings |= ImGui::SliderInt("Confidence Threshold", &shm->m_confidenceThreshold, 0, 255);
    ImGui::Text("Median Filter");
    m_didChangeSettings |= ImGui::RadioButton("None", &shm->m_medianFilter, 0); ImGui::SameLine();
    m_didChangeSettings |= ImGui::RadioButton("3x3", &shm->m_medianFilter, 3); ImGui::SameLine();
    m_didChangeSettings |= ImGui::RadioButton("5x5", &shm->m_medianFilter, 5); ImGui::SameLine();
    m_didChangeSettings |= ImGui::RadioButton("7x7", &shm->m_medianFilter, 7);

    m_didChangeSettings |= ImGui::SliderInt("Bilateral Filter Sigma", &shm->m_bilateralFilterSigma, 0, 65535);
    m_didChangeSettings |= ImGui::Checkbox("L-R Check", &shm->m_enableLRCheck);
    if (shm->m_enableLRCheck) {
      m_didChangeSettings |= ImGui::SliderInt("L-R Check Threshold", &shm->m_leftRightCheckThreshold, 0, 128);
    }
    m_didChangeSettings |= ImGui::Checkbox("Spatial Filter", &shm->m_enableSpatialFilter);
    m_didChangeSettings |= ImGui::Checkbox("Temporal Filter", &shm->m_enableTemporalFilter);
  }
}

void DepthMapGeneratorSHM::internalRenderIMGUIPerformanceGraphs() {
  if (!m_enableProfiling)
    return;

  int plotFlags = ImPlotFlags_NoTitle | ImPlotFlags_NoMouseText | ImPlotFlags_NoInputs | ImPlotFlags_NoMenus | ImPlotFlags_NoBoxSelect;

  if (ImPlot::BeginPlot("##SHMTiming", ImVec2(-1,150), /*flags=*/ plotFlags)) {
      ImPlot::SetupAxis(ImAxis_X1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_NoTickLabels);
      ImPlot::SetupAxisLimits(ImAxis_X1, 0, m_profilingDataBuffer.size(), ImPlotCond_Always);

      ImPlot::SetupAxis(ImAxis_Y1, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_LockMin);
      ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0f, 12.0f, ImPlotCond_Always);

      // Y2 axis is used for the "Missed Deadline" markers
      ImPlot::SetupAxis(ImAxis_Y2, /*label=*/ nullptr, /*flags=*/ ImPlotAxisFlags_Lock | ImPlotAxisFlags_NoDecorations);
      ImPlot::SetupAxisLimits(ImAxis_Y2, 0.0f, 1.0f, ImPlotCond_Always);

      ImPlot::SetupFinish();

      ImPlot::PlotLine("Setup", &m_profilingDataBuffer.data()->m_setupTimeMs, m_profilingDataBuffer.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, m_profilingDataBuffer.offset(), m_profilingDataBuffer.stride());
      ImPlot::PlotLine("Copy",  &m_profilingDataBuffer.data()->m_copyTimeMs,  m_profilingDataBuffer.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, m_profilingDataBuffer.offset(), m_profilingDataBuffer.stride());
      ImPlot::PlotLine("Algo",  &m_profilingDataBuffer.data()->m_algoTimeMs,  m_profilingDataBuffer.size(), /*xscale=*/ 1, /*xstart=*/ 0, /*flags=*/ 0, m_profilingDataBuffer.offset(), m_profilingDataBuffer.stride());

      ImPlot::SetAxis(ImAxis_Y2);
      ImPlot::PlotBars("Missed Deadline", &m_profilingDataBuffer.data()->m_processingTimedOutThisFrame, m_profilingDataBuffer.size(), /*width=*/ 0.7f, /*shift=*/ 0, /*flags=*/ 0, m_profilingDataBuffer.offset(), m_profilingDataBuffer.stride());

      ImPlot::EndPlot();
  }
}

#endif // HAVE_OPENCV_CUDA

