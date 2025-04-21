#pragma once
#include "common/DepthMapGenerator.h"
#include "common/ScrollingBuffer.h"
#include "SingleEyeFitter/SingleEyeFitter.h"
#include <boost/atomic.hpp>
#include <boost/thread.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "rhi/cuda/CudaUtil.h"
#include <cuda.h>
#include <nppcore.h>
#include <nppdefs.h>
#include <NvInfer.h>
#include <algorithm>
#include <memory>
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
    boost::atomic<CaptureBuffer*> m_captureBufferMailbox { nullptr };

  };

  enum CalibrationState {
    kWaitingForValidFrames,
    kCentering,
    kCalibrated
  };

  struct ProcessingState {
    uint64_t m_lastProcessingTimeNs = 0;

    // Calibration state machine data

    uint32_t m_contiguousValidFrameCounter = 0;
    uint32_t m_contiguousInvalidFrameCounter = 0;

    ScrollingBuffer<cv::RotatedRect> m_calibrationSamples {24};
    cv::RotatedRect m_centerCalibrationSample;
    float m_centerPitchDeg = 0.0f;
    float m_centerYawDeg = 0.0f;

    CalibrationState m_calibrationState = kWaitingForValidFrames;



    bool m_requiresPostProcessing = false; // True when we have launched CUDA ops that will update m_trtOutputHostPtr with new data

    ~ProcessingState() {
      releaseResources();
    }

    void releaseResources() {
      m_segmentationExec.reset(nullptr);
      m_roiExec.reset(nullptr);

      CUDA_SAFE_FREE_HOST(m_segInputTensorPtr);
      CUDA_SAFE_FREE_HOST(m_segOutputTensorPtr);

      CUDA_SAFE_FREE_HOST(m_pupilMask1);
      CUDA_SAFE_FREE_HOST(m_pupilMask2);

      for (size_t bufIdx = 0; bufIdx < 2; ++bufIdx) {
        m_captureBuffers[bufIdx].reset(nullptr);
      }

      if (m_frameProcessingGraphExec) {
        cuGraphExecDestroy(m_frameProcessingGraphExec);
        m_frameProcessingGraphExec = nullptr;
      }
      if (m_frameProcessingGraph) {
        cuGraphDestroy(m_frameProcessingGraph);
        m_frameProcessingGraph = nullptr;
      }
    }

    // Center offset of the camera capture, used for translating between the eye-fitter
    // coordinate system (zero at center) and the capture/image coordinate system (zero at left-top)
    cv::Point2f m_captureCenterOffset;

    // ROI scale output
    cv::Mat m_roiScaleMat;

    // Offset from segmentation ROI coordinates to capture mat coordinates. Always positive.
    cv::Point2i m_lastSegROIToCaptureMatOffset;

    // Host/device shared allocation for ROI prediction input. Can be fp16 or int8.
    void* m_roiInputTensorPtr = nullptr;

    // Host/device shared allocation for ROI prediction output. Can be fp16 or int8.
    void* m_roiOutputTensorPtr = nullptr;

    // Host/device shared input to segmentation network -- fp16, -1...1 range.
    _Float16* m_segInputTensorPtr = nullptr;
    size_t m_segInputTensorStrideElements = 0;

    // Host/device shared allocation for TRT output
    _Float16* m_segOutputTensorPtr = nullptr;

    // Host/device shared allocations for pupil mask processing
    uint8_t* m_pupilMask1 = nullptr;
    uint8_t* m_pupilMask2 = nullptr;

    // Postprocessing output
    cv::RotatedRect m_pupilEllipse;
    bool m_eyeFitterOutputsValid = false;
    singleeyefitter::Circle3D<double> m_fitPupilCircle;
    float m_pupilRawPitchDeg = 0.0f, m_pupilRawYawDeg = 0.0f;

    std::vector<cv::RotatedRect> m_eyeFitterSamples;

    // Eye fitter
    singleeyefitter::EyeModelFitter m_eyeModelFitter;
    double m_focalLength = 4.0; // mm
    double m_mm2px_scaling = 0;
    const double pupilRadius() { return 2.0 * m_mm2px_scaling; }
    const double initialEyeZ() { return 50.0 * m_mm2px_scaling; }

    // CUDA graph capture of the tracking model run
    // Internally the TensorRT engine launches several dozen kernels, so capturing
    // that in a graph saves a significant amount of per-frame CPU overhead from kernel launches.
    CUgraph m_frameProcessingGraph = nullptr;
    CUgraphExec m_frameProcessingGraphExec = nullptr;

    // Execution context for running the tracking model
    std::unique_ptr<nvinfer1::IExecutionContext> m_segmentationExec;
    std::unique_ptr<nvinfer1::IExecutionContext> m_roiExec;


    // We store two capture buffers to ensure that we don't lose the most recent buffer when swapping with the capture thread.
    // The capture thread's buffer is not guaranteed to be newer than what we're currently holding.
    std::unique_ptr<CaptureBuffer> m_captureBuffers[2];

    uint64_t captureBufferTimestamp(size_t idx) const { return m_captureBuffers[idx] ? m_captureBuffers[idx]->timestamp : 0; }
    size_t olderCaptureBufferIdx() { return (captureBufferTimestamp(0) < captureBufferTimestamp(1)) ? 0 : 1; }
    size_t newerCaptureBufferIdx() { return (captureBufferTimestamp(0) < captureBufferTimestamp(1)) ? 1 : 0; }
    uint64_t newerCaptureBufferTimestamp() { return std::max<uint64_t>(captureBufferTimestamp(0), captureBufferTimestamp(1)); }

    // Debug view support
    cv::Point2i m_debugLastPredictedROICenter; // in m_debugView coordinates
    cv::Rect m_debugLastSegmentationROI; // in m_debugView coordinates
    cv::Mat m_debugViewGrey; // Full copy of the capture mat, not circle-cropped
    cv::Mat m_debugViewRGB; // Cached allocation for RGB version of m_debugViewGrey
  };


  // per-eye state
  CaptureState m_captureState[2];
  ProcessingState m_processingState[2];

protected:

  void postprocessOneEye(size_t eyeIdx);
  bool postprocessOneEye_fitEllipse(size_t eyeIdx);

  // Low-priority CUDA stream
  CUstream m_cuStream;
  cv::cuda::Stream m_cvStream = cv::cuda::Stream::Null(); // filled in during constructor
  NppStreamContext m_nppContext;

  // Segmentation model I/O sizes
  uint32_t m_segInputWidth = 0, m_segInputHeight = 0;

  size_t m_segOutputSizeBytes = 0;
  size_t m_segOutputRowPitchElements = 0;
  size_t m_segOutputPlanePitchElements = 0;

  // ROI model I/O sizes
  bool m_roiIOIsInt8 = false;
  size_t roiElementSize() const { return m_roiIOIsInt8 ? 1 : 2; }
  uint32_t m_roiInputWidth = 0, m_roiInputHeight = 0;
  uint32_t m_roiOutputWidth = 0, m_roiOutputHeight = 0;
  size_t m_roiOutputSizeBytes = 0;

  // Sync/profiling events
  CUevent m_frameProcessingStartEvent;
  CUevent m_framePostProcessingStartEvent;
  CUevent m_frameProcessingEndEvent;

  // Shared data
  std::unique_ptr<InferLogger> m_logger;
  std::unique_ptr<nvinfer1::IRuntime> m_inferRuntime;
  std::unique_ptr<nvinfer1::ICudaEngine> m_segmentationEngine;
  // names are owned by m_segmentationEngine, don't delete
  const char* m_segInputTensorName = nullptr;
  const char* m_segOutputTensorName = nullptr;

  std::unique_ptr<nvinfer1::ICudaEngine> m_roiEngine;
  // names are owned by m_roiEngine, don't delete
  const char* m_roiInputTensorName = nullptr;
  const char* m_roiOutputTensorName = nullptr;
};

