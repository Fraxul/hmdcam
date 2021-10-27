#include <stdio.h>
#include <signal.h>

#include "common/SHMSegment.h"
#include "common/DepthMapSHM.h"

#include <opencv2/core.hpp>
#include "depthai/depthai.hpp"

#include <time.h>
#include <chrono>

static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}


static SHMSegment<DepthMapSHM>* shm;


struct PerViewData {
  PerViewData() : m_lastSettingsGeneration(0xffffffff), m_paddedWidth(0), m_pipelineWidth(0), m_pipelineHeight(0) {}
  unsigned int m_lastSettingsGeneration;
  unsigned int m_paddedWidth;

  unsigned int m_pipelineWidth, m_pipelineHeight;


  std::shared_ptr<dai::Device> device;
  std::shared_ptr<dai::DataInputQueue> leftQueue, rightQueue, configQueue;
  std::shared_ptr<dai::DataOutputQueue> dispQueue;
};

int main(int argc, char* argv[]) {
  int ppid = getppid();

  shm = SHMSegment<DepthMapSHM>::openSegment("depth-worker");
  if (!shm) {
    printf("dgpu-worker: Unable to open SHM segment\n");
    return -1;
  }
  PerViewData perViewData[DepthMapSHM::maxViews];

  // Signal readyness
  sem_post(&shm->segment()->m_workerReadySem);

  uint32_t frameSequenceNumber = 0;

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

    // Check and update configuration on active views
    for (size_t viewIdx = 0; viewIdx < shm->segment()->m_activeViewCount; ++viewIdx) {
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];
      PerViewData& viewData = perViewData[viewIdx];
      if (viewData.m_lastSettingsGeneration == shm->segment()->m_settingsGeneration)
        continue;

      if (!viewData.device || (viewData.m_pipelineWidth != vp.width) || (viewData.m_pipelineHeight != vp.height)) {
        // Newly attached stream, or the pipeline dimensions have changed.

        // Detach any previously attached device
        viewData.device.reset();


        // Build pipeline components
        dai::Pipeline pipeline;

        viewData.m_pipelineWidth = vp.width;
        viewData.m_pipelineHeight = vp.height;
        viewData.m_paddedWidth = (vp.width + 15) & (~(15UL)); // Pad width up to 16px boundary

        printf("Creating pipeline for stream %zu: %u x %u (padded: %u x %u)\n", viewIdx, vp.width, vp.height, viewData.m_paddedWidth, vp.height);

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

        stereo->setLeftRightCheck(false);
        stereo->setExtendedDisparity(false);
        stereo->setSubpixel(false);

        stereo->setRuntimeModeSwitch(true); // allocate extra resources for runtime mode switching (lr-check, extended, subpixel)

        // link queues to processing node
        inLeft->out.link(stereo->left);
        inRight->out.link(stereo->right);
        inConfig->out.link(stereo->inputConfig);

        stereo->disparity.link(outDisparity->input);

        // Start the pipeline on the first available device
        viewData.device = std::make_shared<dai::Device>(pipeline);
        printf("Stream %zu processing started on device MXID=%s\n", viewIdx, viewData.device->getMxId().c_str());

        // Wire up the device queues
        viewData.leftQueue   =  viewData.device->getInputQueue("left",   /*maxSize=*/ 8, /*blocking=*/ false);
        viewData.rightQueue  =  viewData.device->getInputQueue("right",  /*maxSize=*/ 8, /*blocking=*/ false);
        viewData.configQueue =  viewData.device->getInputQueue("config", /*maxSize=*/ 4, /*blocking=*/ false);
        viewData.dispQueue   =  viewData.device->getOutputQueue("disp",  /*maxSize=*/ 8, /*blocking=*/ false);

      } // pipeline creation

      // Update config
      dai::StereoDepthConfig config;
      config.setConfidenceThreshold(shm->segment()->m_confidenceThreshold);
      config.setMedianFilter((dai::MedianFilter) shm->segment()->m_medianFilter);
      config.setBilateralFilterSigma(shm->segment()->m_bilateralFilterSigma);
      config.setLeftRightCheckThreshold(shm->segment()->m_leftRightCheckThreshold);
      config.setLeftRightCheck(shm->segment()->m_enableLRCheck);

      viewData.configQueue->send(config);

      viewData.m_lastSettingsGeneration = shm->segment()->m_settingsGeneration;
    }

    auto submitTime = std::chrono::steady_clock::now();

    // Submit input frames
    for (size_t viewIdx = 0; viewIdx < shm->segment()->m_activeViewCount; ++viewIdx) {
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];
      PerViewData& viewData = perViewData[viewIdx];

      //cv::Mat leftMat(vp.height, vp.width, CV_8UC1, shm->segment()->data() + vp.inputLeftOffset, vp.inputPitchBytes);
      //cv::Mat rightMat(vp.height, vp.width, CV_8UC1, shm->segment()->data() + vp.inputRightOffset, vp.inputPitchBytes);

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
    } // View loop


    // Download all results
    for (size_t viewIdx = 0; viewIdx < shm->segment()->m_activeViewCount; ++viewIdx) {
      PerViewData& viewData = perViewData[viewIdx];
      DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];

      // StereoBM uses CV_8UC1 disparity map, everything else uses CV_16SC1
      size_t bytesPerPixel = 1; // TODO support 16bpp

      auto frame = viewData.dispQueue->get<dai::ImgFrame>();
      assert(frame->getType() == dai::ImgFrame::Type::RAW8 || frame->getType() == dai::ImgFrame::Type::GRAY8);
      assert(frame->getData().size() == frame->getWidth() * frame->getHeight());
      assert(frame->getWidth() == viewData.m_paddedWidth);
      assert(frame->getHeight() == vp.height);

      for (size_t row = 0; row < vp.height; ++row) {
        memcpy((shm->segment()->data() + vp.outputOffset) + (vp.outputPitchBytes * row), frame->getData().data() + (viewData.m_paddedWidth * bytesPerPixel * row), (vp.width * bytesPerPixel));
      }
    }

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


