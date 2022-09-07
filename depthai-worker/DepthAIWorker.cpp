#include <stdio.h>
#include <signal.h>

#include "common/SHMSegment.h"
#include "common/DepthMapSHM.h"

#include <opencv2/core.hpp>
#include "depthai/depthai.hpp"

#include <time.h>
#include <chrono>
#include <thread>
#include <boost/thread/barrier.hpp>
#include <pthread.h>

static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}


struct PerViewData {
  PerViewData() : m_lastSettingsGeneration(0xffffffff), m_paddedWidth(0), m_pipelineWidth(0), m_pipelineHeight(0) {}
  unsigned int m_lastSettingsGeneration;
  unsigned int m_paddedWidth;

  unsigned int m_pipelineWidth, m_pipelineHeight;


  std::thread processingThread;

  std::shared_ptr<dai::Device> device;
  std::shared_ptr<dai::DataInputQueue> leftQueue, rightQueue, configQueue;
  std::shared_ptr<dai::DataOutputQueue> dispQueue;
};

static SHMSegment<DepthMapSHM>* shm;
PerViewData perViewData[DepthMapSHM::maxViews];
uint32_t frameSequenceNumber = 0;
boost::barrier frameStartBarrier(DepthMapSHM::maxViews + 1);
boost::barrier frameEndBarrier(DepthMapSHM::maxViews + 1);
std::mutex deviceBootMutex;


void viewProcessingThread(size_t viewIdx) {
  DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];
  PerViewData& viewData = perViewData[viewIdx];
  std::shared_ptr<dai::RawStereoDepthConfig> rawStereoConfig = std::make_shared<dai::RawStereoDepthConfig>();

  while (true) {

    // Wait for frame start
    frameStartBarrier.wait();

    // Make sure view is active
    if (viewIdx < shm->segment()->m_activeViewCount) {

      if (!viewData.device || (viewData.m_pipelineWidth != vp.width) || (viewData.m_pipelineHeight != vp.height)) {
        // Newly attached stream, or the pipeline dimensions have changed.

        // Detach any previously attached device
        viewData.device.reset();

        // Build pipeline components
        dai::Pipeline pipeline;

        viewData.m_pipelineWidth = vp.width;
        viewData.m_pipelineHeight = vp.height;
        viewData.m_paddedWidth = (vp.width + 15) & (~(15UL)); // Pad width up to 16px boundary

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
        *rawStereoConfig = stereo->initialConfig.get();

        // link queues to processing node
        inLeft->out.link(stereo->left);
        inRight->out.link(stereo->right);
        inConfig->out.link(stereo->inputConfig);

        stereo->disparity.link(outDisparity->input);

        // Start the pipeline on the first available device
        {
          // Synchronize access to the global device list, so two threads don't pick and try to initialize the same device
          std::lock_guard<std::mutex> g(deviceBootMutex);
          viewData.device = std::make_shared<dai::Device>(pipeline);
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
        rawStereoConfig->costMatching.confidenceThreshold = shm->segment()->m_confidenceThreshold;
        rawStereoConfig->postProcessing.median = ((dai::MedianFilter) shm->segment()->m_medianFilter);
        rawStereoConfig->postProcessing.bilateralSigmaValue = shm->segment()->m_bilateralFilterSigma;
        rawStereoConfig->algorithmControl.leftRightCheckThreshold = shm->segment()->m_leftRightCheckThreshold;
        rawStereoConfig->algorithmControl.enableLeftRightCheck = shm->segment()->m_enableLRCheck;
        rawStereoConfig->postProcessing.spatialFilter.enable = shm->segment()->m_enableSpatialFilter;
        rawStereoConfig->postProcessing.temporalFilter.enable = shm->segment()->m_enableTemporalFilter;

        viewData.configQueue->send(std::make_shared<dai::StereoDepthConfig>(rawStereoConfig));

        viewData.m_lastSettingsGeneration = shm->segment()->m_settingsGeneration;
      }

      // Submit input frames
      {
        auto submitTime = std::chrono::steady_clock::now();
        dai::ImgFrame l, r;
        {
          l.getData().resize(viewData.m_paddedWidth * vp.height);
          unsigned char* outBase = l.getData().data();
          const char* inBase = shm->segment()->data() + vp.inputLeftOffset;
          for (size_t row = 0; row < vp.height; ++row)
            memcpy(outBase + (row * viewData.m_paddedWidth), inBase + (row * vp.inputPitchBytes), vp.width);
        }
        l.setWidth(viewData.m_paddedWidth);
        l.setHeight(vp.height);
        l.setType(dai::ImgFrame::Type::GRAY8);
        l.setInstanceNum((unsigned int) dai::CameraBoardSocket::LEFT);
        l.setSequenceNum(frameSequenceNumber);
        l.setTimestamp(submitTime);

        {
          r.getData().resize(viewData.m_paddedWidth * vp.height);
          unsigned char* outBase = r.getData().data();
          const char* inBase = shm->segment()->data() + vp.inputRightOffset;
          for (size_t row = 0; row < vp.height; ++row)
            memcpy(outBase + (row * viewData.m_paddedWidth), inBase + (row * vp.inputPitchBytes), vp.width);
        }

        r.setWidth(viewData.m_paddedWidth);
        r.setHeight(vp.height);
        r.setType(dai::ImgFrame::Type::GRAY8);
        r.setInstanceNum((unsigned int) dai::CameraBoardSocket::RIGHT);
        r.setSequenceNum(frameSequenceNumber);
        r.setTimestamp(submitTime);

        viewData.leftQueue->send(l);
        viewData.rightQueue->send(r);
      }

      // Download results
      {
        PerViewData& viewData = perViewData[viewIdx];
        DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];

        auto frame = viewData.dispQueue->get<dai::ImgFrame>();
        // Frames are RAW8 unless subpixel is enabled, then they'll be RAW16
        size_t bytesPerPixel = dai::RawImgFrame::typeToBpp(frame->getType());

        assert(frame->getType() == dai::ImgFrame::Type::RAW8 || frame->getType() == dai::ImgFrame::Type::GRAY8);
        assert(frame->getData().size() == frame->getWidth() * frame->getHeight());
        assert(frame->getWidth() == viewData.m_paddedWidth);
        assert(frame->getHeight() == vp.height);

        for (size_t row = 0; row < vp.height; ++row) {
          memcpy((shm->segment()->data() + vp.outputOffset) + (vp.outputPitchBytes * row), frame->getData().data() + (viewData.m_paddedWidth * bytesPerPixel * row), (vp.width * bytesPerPixel));
        }
      }
    } else {
      // ensure device is released
      if (viewData.device) {
        viewData.device.reset();
        viewData.leftQueue.reset();
        viewData.rightQueue.reset();
        viewData.configQueue.reset();
        viewData.dispQueue.reset();
      }
    }

    // Signal completion
    frameEndBarrier.wait();

  } // frame loop
}


int main(int argc, char* argv[]) {
  // Set thread affinity.
  // On Tegra, we get a small but noticeable performance improvement by pinning this application to CPU0-1, and the parent hmdcam application to all other CPUs.
  // I assume this is because CPU0 handles interrupts for the XHCI driver; we want to try and limit its workload to mostly USB-related things to reduce latency.
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


  int ppid = getppid();

  shm = SHMSegment<DepthMapSHM>::openSegment("depth-worker");
  if (!shm) {
    printf("depthai-worker: Unable to open SHM segment\n");
    return -1;
  }

  // Launch processing threads
  for (size_t viewIdx = 0; viewIdx < DepthMapSHM::maxViews; ++viewIdx) {
    perViewData[viewIdx].processingThread = std::thread(viewProcessingThread, viewIdx);
  }

  // Signal readyness
  sem_post(&shm->segment()->m_workerReadySem);

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

    uint64_t startTime = currentTimeNs();

    // Start processing threads
    frameStartBarrier.wait();

    // Wait for processing threads to finish
    frameEndBarrier.wait();

    for (size_t viewIdx = 0; viewIdx < shm->segment()->m_activeViewCount; ++viewIdx) {
      //PerViewData& viewData = perViewData[viewIdx];
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];

      shm->flush(vp.outputOffset, vp.outputPitchBytes * vp.height);
    }

    uint64_t deltaT = currentTimeNs() - startTime;
    shm->segment()->m_frameTimeMs = static_cast<double>(deltaT) / 1000000.0;

    // Finished processing all views -- signal completion
    sem_post(&shm->segment()->m_workFinishedSem);

    ++frameSequenceNumber;
  } // Work loop

  return 0;
}


