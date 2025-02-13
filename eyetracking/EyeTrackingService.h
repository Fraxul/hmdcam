#pragma once
#include "common/DepthMapGenerator.h"
#include "common/ScrollingBuffer.h"
#include "SingleEyeFitter/SingleEyeFitter.h"
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "rhi/cuda/CudaUtil.h"
#include <cuda.h>
#include <nppcore.h>
#include <nppdefs.h>
#include <NvInfer.h>
#include <algorithm>
#include <vector>

class InferLogger;

#define PER_EYE for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)

class EyeTrackingService {
public:
  EyeTrackingService();
  ~EyeTrackingService();

  void internalLoadSettings(cv::FileStorage& fs);
  void internalSaveSettings(cv::FileStorage& fs);

  void setInputFilename(size_t eyeIdx, const std::string& s) {
    assert(eyeIdx < 2);
    m_captureState[eyeIdx].m_inputFilename = s;
  }

  bool processFrame();

  float m_lastFrameProcessingTimeMs = 0.0f;
  float m_lastFramePostProcessingTimeMs = 0.0f;

  cv::Mat& getDebugViewForEye(size_t eyeIdx);

  struct CaptureBuffer {
    cv::Mat mat;
    uint64_t timestamp; // currentTimeNs
  };

  struct CaptureState {
    std::string m_inputFilename;


    // Worker thread entrypoint
    void captureWorkerThread();

    // Thread that is reading/decoding frames from the cv::VideoCapture object
    boost::thread m_captureThread;

    bool m_captureThreadAlive = false;

    // Ratelimiting for capture-open attempts
    uint64_t m_lastCaptureOpenAttemptTimeNs = 0; // currentTimeNs

    // OpenCV video capture object
    cv::VideoCapture m_capture;

    // Latest buffer filled by the capture worker will be in here.
    // Main thread will swap it with its previous buffer (or NULL if it didn't have one)
    // Capture thread will swap its just-filled buffer with whatever's in here, and allocate a new one if it gets back NULL
    // We should end up with 3 buffers in-flight between the capture thread, main thread, and mailbox.
    boost::atomic<CaptureBuffer*> m_captureBufferMailbox;

  };

  struct ProcessingState {
    uint64_t m_lastProcessingTimeNs = 0;

    bool m_requiresPostProcessing = false; // True when we have launched CUDA ops that will update m_trtOutputHostPtr with new data

    ~ProcessingState() {
      releaseResources();
    }

    void releaseResources() {
      if (m_exec) {
        delete m_exec;
        m_exec = nullptr;
      }

      CUDA_SAFE_FREE(m_trtInputMatPtr);
      CUDA_SAFE_FREE_HOST(m_trtOutputHostPtr);

      CUDA_SAFE_FREE_HOST(m_classIndex);
      CUDA_SAFE_FREE_HOST(m_pupilMask1);
      CUDA_SAFE_FREE_HOST(m_pupilMask2);

      for (size_t bufIdx = 0; bufIdx < 2; ++bufIdx) {
        if (m_captureBuffers[bufIdx]) {
          delete m_captureBuffers[bufIdx];
          m_captureBuffers[bufIdx] = nullptr;
        }
      }

      if (m_frameProcessingGraphExec) {
        cuGraphExecDestroy(m_frameProcessingGraphExec);
        m_frameProcessingGraphExec = nullptr;
      }
      if (m_frameProcessingGraph) {
        cuGraphDestroy(m_frameProcessingGraph);
        m_frameProcessingGraph = nullptr;
      }

      m_trtInputMat = cv::cuda::GpuMat();
    }

    // Input to TensorRT -- fp16, -1...1 range
    cv::cuda::GpuMat m_trtInputMat;

    // Explicit allocations for TensorRT GpuMats, since we need control over the pitch
    CUdeviceptr m_trtInputMatPtr = 0;

    // Host/device shared allocation for TRT output
    _Float16* m_trtOutputHostPtr = 0;

    // Host/device shared allocation for class index
    uint8_t* m_classIndex = 0;

    // Host/device shared allocations for pupil mask processing
    uint8_t* m_pupilMask1 = 0;
    uint8_t* m_pupilMask2 = 0;

    // Upload of capture buffer
    cv::cuda::GpuMat m_preWarpGpuMat;
    cv::cuda::GpuMat m_preHistEqMat;

    cv::cuda::GpuMat m_postHistEqMat;

    // CLAHE processor
    cv::Ptr<cv::cuda::CLAHE> m_clahe;

    // Postprocessing output
    cv::RotatedRect m_pupilEllipse;
    float m_pupilContourArea = 0;

    std::vector<cv::RotatedRect> m_eyeFitterSamples;

    // Eye fitter
    singleeyefitter::EyeModelFitter m_eyeModelFitter;
    double m_focalLength = 6.0; // mm
    double m_mm2px_scaling;
    const double pupilRadius() { return 2.0 * m_mm2px_scaling; }
    const double initialEyeZ() { return 100.0 * m_mm2px_scaling; }

    // CUDA graph capture of the tracking model run
    // Internally the TensorRT engine launches several dozen kernels, so capturing
    // that in a graph saves a significant amount of per-frame CPU overhead from kernel launches.
    CUgraph m_frameProcessingGraph = nullptr;
    CUgraphExec m_frameProcessingGraphExec = nullptr;

    // Execution context for running the tracking model
    nvinfer1::IExecutionContext* m_exec = nullptr;


    // We store two capture buffers to ensure that we don't lose the most recent buffer when swapping with the capture thread.
    // The capture thread's buffer is not guaranteed to be newer than what we're currently holding.
    CaptureBuffer* m_captureBuffers[2] = {nullptr, nullptr};

    uint64_t captureBufferTimestamp(size_t idx) const { return (m_captureBuffers[idx] == nullptr) ? 0 : m_captureBuffers[idx]->timestamp; }
    size_t olderCaptureBufferIdx() { return (captureBufferTimestamp(0) < captureBufferTimestamp(1)) ? 0 : 1; }
    size_t newerCaptureBufferIdx() { return (captureBufferTimestamp(0) < captureBufferTimestamp(1)) ? 1 : 0; }
    uint64_t newerCaptureBufferTimestamp() { return std::max<uint64_t>(captureBufferTimestamp(0), captureBufferTimestamp(1)); }

    // Debug view support
    cv::Mat m_debugViewGrey;
    cv::Mat m_debugViewRGB;
  };


  // per-eye state
  CaptureState m_captureState[2];
  ProcessingState m_processingState[2];

protected:
  size_t m_currentlyProcessingEyeIdx = 0;

  // Low-priority CUDA stream
  CUstream m_cuStream;
  cv::cuda::Stream m_cvStream = cv::cuda::Stream::Null(); // filled in during constructor
  NppStreamContext m_nppContext;

  // Input image size
  uint32_t m_trtInputWidth = 0, m_trtInputHeight = 0;

  size_t m_trtOutputSizeBytes = 0;
  size_t m_trtOutputRowPitchElements = 0;
  size_t m_trtOutputPlanePitchElements = 0;

  // Profiling events
  bool m_enableProfiling = true;
  CUevent m_frameProcessingStartEvent;
  CUevent m_framePostProcessingStartEvent;
  CUevent m_frameProcessingEndEvent;

  // Shared data
  InferLogger* m_logger = nullptr;
  nvinfer1::IRuntime* m_inferRuntime = nullptr;
  nvinfer1::ICudaEngine* m_inferEngine = nullptr;

  // Input LUT
  CUdeviceptr m_inputLUT = 0;

};

