#include <stdio.h>
#include <signal.h>

#include "common/SHMSegment.h"
#include "common/DepthMapSHM.h"
#include "common/Timing.h"

#include <opencv2/core.hpp>
#include "depthai/depthai.hpp"

#include <time.h>
#include <chrono>
#include <pthread.h>
#include "nvToolsExt.h"

struct PerViewData {
  unsigned int m_lastSettingsGeneration = 0xffffffff;
  unsigned int m_paddedWidth = 0;

  unsigned int m_pipelineWidth = 0, m_pipelineHeight = 0;

  std::shared_ptr<dai::Device> device;
  std::shared_ptr<dai::DataInputQueue> leftQueue, rightQueue, configQueue;
  std::shared_ptr<dai::DataOutputQueue> dispQueue;

  std::shared_ptr<dai::RawStereoDepthConfig> rawStereoConfig = std::make_shared<dai::RawStereoDepthConfig>();
  dai::ImgFrame inputFrame[2];
};

int main(int argc, char* argv[]) {


  // Set thread affinity.
  // On Tegra, we get a small but noticeable performance improvement by pinning this application to CPU0-1, and the parent hmdcam application to all other CPUs.
  // I assume this is because CPU0 handles interrupts for the XHCI driver; we want to try and limit its workload to mostly USB-related things to reduce latency.
#if 0
  {
    cpu_set_t cpuset;
    // Create affinity mask for CPU0-1
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    CPU_SET(1, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
      perror("pthread_setaffinity");
    }
  }
#endif

  SHMSegment<DepthMapSHM>* shm = SHMSegment<DepthMapSHM>::openSegment("depth-worker");
  if (!shm) {
    printf("depthai-worker: Unable to open SHM segment\n");
    return -1;
  }

  // Signal readyness
  sem_post(&shm->segment()->m_workerReadySem);

  std::vector<PerViewData> perViewData;
  perViewData.reserve(DepthMapSHM::maxViews);

  uint32_t frameSequenceNumber = 0;
  int ppid = getppid();

  while (true) {
    {
      struct timespec ts;
      ts.tv_sec = 1;
      ts.tv_nsec = 0;
      if (sem_timedwait(&shm->segment()->m_workAvailableSem, &ts) < 0) {
        if (errno == ETIMEDOUT) {
          if (kill(ppid, 0) != 0) {
            printf("depthai-worker: parent process %d has exited\n", ppid);
            return 0;
          }
          continue;
        } else {
          perror("sem_timedwait");
          return -1;
        }
      }
    }
    nvtxMarkA("Frame Start");

    uint64_t startTime = currentTimeNs();

    // Trim view list first -- if it shrunk, this will release devices.
    perViewData.resize(shm->segment()->m_activeViewCount);

    // Per-view initialization
    for (size_t viewIdx = 0; viewIdx < perViewData.size(); ++viewIdx) {
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];
      PerViewData& viewData = perViewData[viewIdx];

      if (!viewData.device || (viewData.m_pipelineWidth != vp.width) || (viewData.m_pipelineHeight != vp.height)) {
        // Newly attached stream, or the pipeline dimensions have changed.

        // Detach any previously attached device
        viewData.device.reset();

        // Build pipeline components
        dai::Pipeline pipeline;

        viewData.m_pipelineWidth = vp.width;
        viewData.m_pipelineHeight = vp.height;
        viewData.m_paddedWidth = (vp.width + 15) & (~(15UL)); // Pad width up to 16px boundary

        viewData.inputFrame[0].getData().resize(viewData.m_paddedWidth * vp.height);
        viewData.inputFrame[0].setWidth(viewData.m_paddedWidth);
        viewData.inputFrame[0].setHeight(vp.height);
        viewData.inputFrame[0].setType(dai::ImgFrame::Type::GRAY8);
        viewData.inputFrame[0].setInstanceNum((unsigned int) dai::CameraBoardSocket::LEFT);

        viewData.inputFrame[1].getData().resize(viewData.m_paddedWidth * vp.height);
        viewData.inputFrame[1].setWidth(viewData.m_paddedWidth);
        viewData.inputFrame[1].setHeight(vp.height);
        viewData.inputFrame[1].setType(dai::ImgFrame::Type::GRAY8);
        viewData.inputFrame[1].setInstanceNum((unsigned int) dai::CameraBoardSocket::RIGHT);

        printf("[%zu] Creating stream pipeline: %u x %u (padded: %u x %u)\n", viewIdx, vp.width, vp.height, viewData.m_paddedWidth, vp.height);

        auto inLeft = pipeline.create<dai::node::XLinkIn>();
        auto inRight = pipeline.create<dai::node::XLinkIn>();
        auto inConfig = pipeline.create<dai::node::XLinkIn>();
        auto stereo = pipeline.create<dai::node::StereoDepth>();
        auto outDisparity = pipeline.create<dai::node::XLinkOut>();

        inLeft->setStreamName("left");
        inRight->setStreamName("right");
        inConfig->setStreamName("config");
        outDisparity->setStreamName("disp");

        // StereoDepth config
        stereo->initialConfig.setConfidenceThreshold(245);
        stereo->setRectification(false); // inputs are already rectified
        stereo->setRectifyEdgeFillColor(0);  // black, to better see the cutout
        stereo->setInputResolution(viewData.m_paddedWidth, vp.height);
        stereo->initialConfig.setMedianFilter(dai::MedianFilter::KERNEL_5x5);

        stereo->setDepthAlign(dai::StereoDepthConfig::AlgorithmControl::DepthAlign::RECTIFIED_LEFT);
        stereo->setLeftRightCheck(false);
        stereo->setExtendedDisparity(false);
        stereo->setSubpixel(false);

        stereo->setRuntimeModeSwitch(true); // allocate extra resources for runtime mode switching (lr-check, extended, subpixel)

        // cache initial raw config
        *viewData.rawStereoConfig = stereo->initialConfig.get();

        // link queues to processing node
        inLeft->out.link(stereo->left);
        inRight->out.link(stereo->right);
        inConfig->out.link(stereo->inputConfig);

        stereo->disparity.link(outDisparity->input);

        // Start the pipeline on the first available device
        viewData.device = std::make_shared<dai::Device>(pipeline);

        printf("[%zu] Default XLink chunk size: %u\n", viewIdx, viewData.device->getXLinkChunkSize());
        unsigned int chunkSize = viewData.device->getXLinkChunkSize();
        {
          char* e = getenv("XLINK_CHUNK_SIZE");
          if (e) {
            chunkSize = atoi(e);
            printf("[%zu] New XLink chunk size: %u\n", viewIdx, chunkSize);
            viewData.device->setXLinkChunkSize(chunkSize);
          }
        }

        printf("[%zu] Processing started on device MXID=%s\n", viewIdx, viewData.device->getMxId().c_str());

        // Wire up the device queues
        viewData.leftQueue   =  viewData.device->getInputQueue("left",   /*maxSize=*/ 8, /*blocking=*/ false);
        viewData.rightQueue  =  viewData.device->getInputQueue("right",  /*maxSize=*/ 8, /*blocking=*/ false);
        viewData.configQueue =  viewData.device->getInputQueue("config", /*maxSize=*/ 4, /*blocking=*/ false);
        viewData.dispQueue   =  viewData.device->getOutputQueue("disp",  /*maxSize=*/ 8, /*blocking=*/ false);

        // Force config update
        viewData.m_lastSettingsGeneration = shm->segment()->m_settingsGeneration - 1;
      } // pipeline creation

      if (viewData.m_lastSettingsGeneration != shm->segment()->m_settingsGeneration) {
        // Update config

        viewData.rawStereoConfig->costMatching.confidenceThreshold = shm->segment()->m_confidenceThreshold;
        viewData.rawStereoConfig->postProcessing.median = ((dai::MedianFilter) shm->segment()->m_medianFilter);
        viewData.rawStereoConfig->postProcessing.bilateralSigmaValue = shm->segment()->m_bilateralFilterSigma;
        viewData.rawStereoConfig->algorithmControl.leftRightCheckThreshold = shm->segment()->m_leftRightCheckThreshold;
        viewData.rawStereoConfig->algorithmControl.enableLeftRightCheck = shm->segment()->m_enableLRCheck;
        viewData.rawStereoConfig->postProcessing.spatialFilter.enable = shm->segment()->m_enableSpatialFilter;
        viewData.rawStereoConfig->postProcessing.temporalFilter.enable = shm->segment()->m_enableTemporalFilter;

        viewData.configQueue->send(std::make_shared<dai::StereoDepthConfig>(viewData.rawStereoConfig));

        viewData.m_lastSettingsGeneration = shm->segment()->m_settingsGeneration;
      }
    } // Per-view initialization

    nvtxMarkA("Upload start");
    // Per-view uploads
    for (size_t viewIdx = 0; viewIdx < perViewData.size(); ++viewIdx) {
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];
      PerViewData& viewData = perViewData[viewIdx];


      // Submit input frames
      auto submitTime = std::chrono::steady_clock::now();

      for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx) {
        viewData.inputFrame[eyeIdx].setSequenceNum(frameSequenceNumber);
        viewData.inputFrame[eyeIdx].setTimestamp(submitTime);

        unsigned char* outBase = viewData.inputFrame[eyeIdx].getData().data();
        const char* inBase = shm->segment()->data() + vp.inputOffset[eyeIdx];
        for (size_t row = 0; row < vp.height; ++row)
          memcpy(outBase + (row * viewData.m_paddedWidth), inBase + (row * vp.inputPitchBytes), vp.width);

      }


      nvtxMarkA("leftQueue send");
      viewData.leftQueue->send(viewData.inputFrame[0]);
      nvtxMarkA("rightQueue send");
      viewData.rightQueue->send(viewData.inputFrame[1]);
    } // Per-view uploads


    nvtxMarkA("Download start");

    for (size_t viewIdx = 0; viewIdx < perViewData.size(); ++viewIdx) {
      // Download results
      PerViewData& viewData = perViewData[viewIdx];
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];

      nvtxMarkA("Download disparity");
      auto frame = viewData.dispQueue->get<dai::ImgFrame>();
      // Frames are RAW8 unless subpixel is enabled, then they'll be RAW16
      size_t bytesPerPixel = dai::RawImgFrame::typeToBpp(frame->getType());

      assert(frame->getType() == dai::ImgFrame::Type::RAW8 || frame->getType() == dai::ImgFrame::Type::GRAY8);
      assert(frame->getData().size() == frame->getWidth() * frame->getHeight());
      assert(frame->getWidth() == viewData.m_paddedWidth);
      assert(frame->getHeight() == vp.height);

      nvtxMarkA("Copy disparity to SHM");
      for (size_t row = 0; row < vp.height; ++row) {
        memcpy((shm->segment()->data() + vp.outputOffset) + (vp.outputPitchBytes * row), frame->getData().data() + (viewData.m_paddedWidth * bytesPerPixel * row), (vp.width * bytesPerPixel));
      }
    }

    for (size_t viewIdx = 0; viewIdx < shm->segment()->m_activeViewCount; ++viewIdx) {
      //PerViewData& viewData = perViewData[viewIdx];
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];

      shm->flush(vp.outputOffset, vp.outputPitchBytes * vp.height);
    }

    shm->segment()->m_frameTimeMs = deltaTimeMs(startTime, currentTimeNs());

    nvtxMarkA("Frame End");
    // Finished processing all views -- signal completion
    sem_post(&shm->segment()->m_workFinishedSem);

    ++frameSequenceNumber;
  } // Work loop

  return 0;
}

