#include <stdio.h>
#include <signal.h>

#include "common/FxThreading.h"
#include "common/SHMSegment.h"
#include "common/DepthMapSHM.h"
#include "common/Timing.h"

#include <opencv2/core.hpp>
#include "depthai/depthai.hpp"
#include "depthai/xlink/XLinkStream.hpp"

#include <time.h>
#include <chrono>
#include <pthread.h>
#include "nvToolsExt.h"

#define PER_EYE for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)

struct PerViewData {
  unsigned int m_lastSettingsGeneration = 0xffffffff;
  unsigned int m_paddedWidth = 0;

  unsigned int m_pipelineWidth = 0, m_pipelineHeight = 0;

  std::shared_ptr<dai::Device> device;
  std::shared_ptr<dai::DataInputQueue> configQueue;
  std::shared_ptr<dai::DataOutputQueue> dispQueue;

  std::shared_ptr<dai::RawStereoDepthConfig> rawStereoConfig = std::make_shared<dai::RawStereoDepthConfig>();

  struct InputFrame {
    InputFrame() : daiRawImgFrame(std::make_shared<dai::RawImgFrame>()), daiFrame(daiRawImgFrame) {}

    std::shared_ptr<dai::DataInputQueue> queue;

    std::shared_ptr<dai::RawImgFrame> daiRawImgFrame;
    dai::ImgFrame daiFrame;

    std::vector<uint8_t> serializedMetadata;
    dai::DatatypeEnum serializedDatatype;
    uint8_t* dmaBuffer = nullptr;
    uint32_t dmaBufferSize = 0;

    void freeDMABuffer() {
      if (dmaBuffer) {
        printf("InputFrame(%p) free dmaBuffer=%p size=%u\n", this, dmaBuffer, dmaBufferSize);
        queue->xlinkStream().deallocateDMABuffer(dmaBuffer, dmaBufferSize);
      }

      dmaBuffer = nullptr;
      dmaBufferSize = 0;
    }
    void ensureDMABuffer(uint32_t targetSize) {
      if (dmaBufferSize >= targetSize)
        return;
      freeDMABuffer();
      dmaBufferSize = (targetSize + 4095) & (~(4095));

      dmaBuffer = queue->xlinkStream().allocateDMABuffer(dmaBufferSize);
      printf("InputFrame(%p) allocate dmaBuffer=%p size=%u (targetSize=%u)\n", this, dmaBuffer, dmaBufferSize, targetSize);
    }
  };
  InputFrame inputFrame[2];
};


SHMSegment<DepthMapSHM>* shm = nullptr;
std::vector<PerViewData> perViewData;
uint32_t frameSequenceNumber = 0;

inline size_t pitchCopy(void* dest, size_t destPitchBytes, const void* src, size_t srcPitchBytes, size_t copyWidthBytes, size_t copyHeightRows) {
  assert(copyWidthBytes <= destPitchBytes && copyWidthBytes <= srcPitchBytes);
  for (size_t row = 0; row < copyHeightRows; ++row) {
    memcpy(reinterpret_cast<uint8_t*>(dest) + (row * destPitchBytes), reinterpret_cast<const uint8_t*>(src) + (row * srcPitchBytes), copyWidthBytes);
  }
  return destPitchBytes * copyHeightRows;
}

void perViewPerEyeUpload(size_t viewEyeIdx);

int main(int argc, char* argv[]) {

  // Set scheduling for worker threads to SCHED_FIFO. This substantially reduces frametime/latency jitter
  // without interfering with the rest of the system too much, since thread runtimes here are generally
  // short -- this process spends most of its time waiting on USB I/O. For safety, though, we do limit
  // it to running on two cores; currently CPU6-CPU7. The parent hmdcam binary masks those CPUs off and
  // does not schedule on them.
  //
  // For this to work correctly, the following sysctl tweak may be required on L4T:
  //   sudo sysctl -w kernel.sched_rt_runtime_us=-1
  //
  // The binary must also be granted the CAP_SYS_NICE capability post-build:
  //   sudo setcap cap_sys_nice+ep build/bin/depthai-worker

  {
    cpu_set_t cpuset;
    // Create affinity mask for CPU6-7
    CPU_ZERO(&cpuset);
    CPU_SET(6, &cpuset);
    CPU_SET(7, &cpuset);

    if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) != 0) {
      perror("pthread_setaffinity");
    }

    struct sched_param p;
    p.sched_priority = 1;
    if (pthread_setschedparam(pthread_self(), SCHED_FIFO, &p) != 0) {
      perror("pthread_setschedparam");
    }
  }

  shm = SHMSegment<DepthMapSHM>::openSegment("depth-worker");
  if (!shm) {
    printf("depthai-worker: Unable to open SHM segment\n");
    return -1;
  }

  // Signal readyness
  sem_post(&shm->segment()->m_workerReadySem);

  FxThreading::detail::init();
  perViewData.reserve(DepthMapSHM::maxViews);

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

        // Setup daiFrame metadata fields
        PER_EYE {
          viewData.inputFrame[eyeIdx].daiFrame.getData().resize(viewData.m_paddedWidth * vp.height);
          viewData.inputFrame[eyeIdx].daiFrame.setWidth(viewData.m_paddedWidth);
          viewData.inputFrame[eyeIdx].daiFrame.setHeight(vp.height);
          viewData.inputFrame[eyeIdx].daiFrame.setType(dai::ImgFrame::Type::GRAY8);
          viewData.inputFrame[eyeIdx].daiFrame.setInstanceNum(eyeIdx == 0 ? ((unsigned int) dai::CameraBoardSocket::LEFT) : ((unsigned int) dai::CameraBoardSocket::RIGHT));
        }

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
        stereo->setSubpixel(true);

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
        viewData.inputFrame[0].queue = viewData.device->getInputQueue("left",   /*maxSize=*/ 8, /*blocking=*/ false);
        viewData.inputFrame[1].queue = viewData.device->getInputQueue("right",  /*maxSize=*/ 8, /*blocking=*/ false);
        viewData.configQueue         = viewData.device->getInputQueue("config", /*maxSize=*/ 4, /*blocking=*/ false);
        viewData.dispQueue           = viewData.device->getOutputQueue("disp",  /*maxSize=*/ 8, /*blocking=*/ false);

        // Stop input threads on L&R queues and take ownership of the underlying XLink streams
        PER_EYE viewData.inputFrame[eyeIdx].queue->takeStreamOwnership();

        // Force config update
        viewData.m_lastSettingsGeneration = shm->segment()->m_settingsGeneration - 1;
      } // pipeline creation

      if (viewData.m_lastSettingsGeneration != shm->segment()->m_settingsGeneration) {
        // Update config

        viewData.rawStereoConfig->algorithmControl.subpixelFractionalBits = shm->segment()->m_subpixelFractionalBits;
        viewData.rawStereoConfig->costMatching.confidenceThreshold = shm->segment()->m_confidenceThreshold;
        if (shm->segment()->m_subpixelFractionalBits == 3) {
          viewData.rawStereoConfig->postProcessing.median = ((dai::MedianFilter) shm->segment()->m_medianFilter);
        } else {
          viewData.rawStereoConfig->postProcessing.median = dai::MedianFilter::MEDIAN_OFF; // only supported for 3-bit subpixel mode
        }
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

    // Run frame loop on each device per-view per-eye
    FxThreading::runArrayTask(0, perViewData.size() * 2, perViewPerEyeUpload);

    // Flush SHM writes
    for (size_t viewIdx = 0; viewIdx < shm->segment()->m_activeViewCount; ++viewIdx) {
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

void perViewPerEyeUpload(size_t viewEyeIdx) {
  size_t viewIdx = viewEyeIdx >> 1;
  size_t eyeIdx = viewEyeIdx & 1;

  DepthMapSHM::ViewParams& vp = shm->segment()->m_viewParams[viewIdx];
  PerViewData& viewData = perViewData[viewIdx];

  // Submit input frames
  auto submitTime = std::chrono::steady_clock::now();

  PerViewData::InputFrame& frameData = viewData.inputFrame[eyeIdx];

  nvtxMarkA(eyeIdx == 0 ? "Left Frame Send" : "Right Frame Send");

  // Update and serialize metadata
  frameData.daiFrame.setSequenceNum(frameSequenceNumber);
  frameData.daiFrame.setTimestamp(submitTime);

  frameData.daiRawImgFrame->serialize(frameData.serializedMetadata, frameData.serializedDatatype);

  // Borrowed from StreamMessageParser::serializeMessage(const RawBuffer&)
  uint32_t serializedSize = frameData.daiFrame.getData().size() + frameData.serializedMetadata.size() + 8;

  // (re)allocate dmaBuffer, if necessary
  frameData.ensureDMABuffer(serializedSize);

  // 4B datatype & 4B metadata size
  std::array<std::uint8_t, 4> leDatatype;
  std::array<std::uint8_t, 4> leMetadataSize;
  for(int i = 0; i < 4; i++) leDatatype[i] = (static_cast<std::int32_t>(frameData.serializedDatatype) >> (i * 8)) & 0xFF;
  uint32_t metadataSize = frameData.serializedMetadata.size();
  for(int i = 0; i < 4; i++) leMetadataSize[i] = (metadataSize >> i * 8) & 0xFF;

  // Copy image data from SHM to DMABUF
  size_t writePtr = 0;
  writePtr += pitchCopy(frameData.dmaBuffer, viewData.m_paddedWidth, shm->segment()->data() + vp.inputOffset[eyeIdx], vp.inputPitchBytes, vp.width, vp.height);
#define WRITE(data, size) memcpy(frameData.dmaBuffer + writePtr, data, size); writePtr += size;
  WRITE(frameData.serializedMetadata.data(), frameData.serializedMetadata.size());
  WRITE(leDatatype.data(), leDatatype.size());
  WRITE(leMetadataSize.data(), leMetadataSize.size());
#undef  WRITE
  frameData.queue->xlinkStream().write(frameData.dmaBuffer, writePtr);


  if (eyeIdx != 0)
    return; // Use eye0 thread to handle the download

  // Download results
  nvtxMarkA("Download disparity");
  auto frame = viewData.dispQueue->get<dai::ImgFrame>();
  // Frames are RAW8 unless subpixel is enabled, then they'll be RAW16
  size_t bytesPerPixel = dai::RawImgFrame::typeToBpp(frame->getType());

  assert(frame->getType() == dai::ImgFrame::Type::RAW16);
  assert(frame->getData().size() == frame->getWidth() * frame->getHeight() * bytesPerPixel);
  assert(frame->getWidth() == viewData.m_paddedWidth);
  assert(frame->getHeight() == vp.height);

  nvtxMarkA("Copy disparity to SHM");
  pitchCopy(shm->segment()->data() + vp.outputOffset, vp.outputPitchBytes, frame->getData().data(), viewData.m_paddedWidth * bytesPerPixel, vp.width * bytesPerPixel, vp.height);
}

