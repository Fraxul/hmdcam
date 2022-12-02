#pragma once
#ifdef HAVE_VPI2
#include "common/DepthMapGenerator.h"
#include "rhi/cuda/CudaUtil.h"
#include "common/VPIUtil.h"
#include <vpi/VPI.h>
#include <vpi/algo/StereoDisparity.h>
#include <vpi/CUDAInterop.h>
#include <cuda.h>
#include <nppcore.h>

#define PER_EYE for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)

class DepthMapGeneratorVPI : public DepthMapGenerator {
public:
  DepthMapGeneratorVPI();
  virtual ~DepthMapGeneratorVPI();

protected:
  virtual void internalLoadSettings(cv::FileStorage&);
  virtual void internalSaveSettings(cv::FileStorage&);
  virtual void internalProcessFrame();
  virtual void internalRenderIMGUI();
  virtual void internalRenderIMGUIPerformanceGraphs();

  struct ViewDataVPI : public ViewData {
    ViewDataVPI() {
      VPI_CHECK(vpiStreamCreateWrapperCUDA((CUstream) m_cuStream.cudaPtr(), /*flags=*/ 0, &m_stream));

      VPI_CHECK(vpiEventCreate(/*flags=*/ 0, &m_stereoStartedEvent));
      VPI_CHECK(vpiEventCreate(/*flags=*/ 0, &m_stereoFinishedEvent));

      // Set up valid initial state for events
      vpiEventRecord(m_stereoStartedEvent, m_stream);
      vpiEventRecord(m_stereoFinishedEvent, m_stream);

      // Init NPP stream context
      memset(&m_nppStreamContext, 0, sizeof(m_nppStreamContext));
      nppSetStream((CUstream) m_cuStream.cudaPtr());
      nppGetStreamContext(&m_nppStreamContext);
    }

    virtual ~ViewDataVPI() {
      releaseVPIResources();

      vpiStreamDestroy(m_stream); m_stream = NULL;
      vpiEventDestroy(m_stereoStartedEvent); m_stereoStartedEvent = NULL;
      vpiEventDestroy(m_stereoFinishedEvent); m_stereoFinishedEvent = NULL;
    }

    void releaseVPIResources() {
      PER_EYE({
        vpiPayloadDestroy(m_remapPayload[eyeIdx]);    m_remapPayload[eyeIdx] = NULL;
        vpiImageDestroy(m_disparityInput[eyeIdx]);    m_disparityInput[eyeIdx] = NULL;
        vpiImageDestroy(m_resizedTransposed[eyeIdx]); m_resizedTransposed[eyeIdx] = NULL;
      });

      vpiPayloadDestroy(m_disparityEstimator); m_disparityEstimator = NULL;

      vpiImageDestroy(m_disparity); m_disparity = NULL;
      vpiImageDestroy(m_confidence); m_confidence = NULL;

      vpiImageDestroy(m_disparityTransposed); m_disparityTransposed = NULL;
      vpiImageDestroy(m_confidenceTransposed); m_confidenceTransposed = NULL;
    }

    cv::cuda::Stream m_cuStream;
    VPIStream m_stream = nullptr;
    NppStreamContext m_nppStreamContext;
    VPIEvent m_stereoStartedEvent = nullptr;
    VPIEvent m_stereoFinishedEvent = nullptr;

    cv::cuda::Event m_cudaSetupStartedEvent;
    cv::cuda::Event m_cudaSetupFinishedEvent;

    float m_cudaSetupTimeMs = 0, m_stereoTimeMs = 0;

    // Remap payloads for rectification
    VPIPayload m_remapPayload[2] = {nullptr, nullptr};
    cv::cuda::GpuMat m_undistortRectifyMap_gpu[2];

    // Output from remap
    cv::cuda::GpuMat m_rectifiedMat[2];

    // Output from resize, wrapped for handoff to VPI
    cv::cuda::GpuMat m_disparityInputMat[2];
    VPIImage m_disparityInput[2] = {nullptr, nullptr};

    VPIPayload m_disparityEstimator = nullptr;

    VPIImage m_disparity = nullptr;
    VPIImage m_confidence = nullptr;

    // Vertical stereo support
    VPIImage m_resizedTransposed[2] = {nullptr, nullptr};
    VPIImage m_disparityTransposed = nullptr;
    VPIImage m_confidenceTransposed = nullptr;
  };

  virtual ViewData* newEmptyViewData() { return new ViewDataVPI(); }
  virtual void internalUpdateViewData();


  CUstream m_masterCUStream = nullptr;
  VPIStream m_masterStream = nullptr;
  VPIEvent m_masterFrameStartEvent = nullptr;
  VPIEvent m_masterFrameFinishedEvent = nullptr;
  float m_frameTimeMs = 0;

  VPIStereoDisparityEstimatorParams m_params;

  bool m_enableProfiling = true;

private:
  ViewDataVPI* viewDataAtIndex(size_t index) { return static_cast<ViewDataVPI*>(m_viewData[index]); }
};
#endif // HAVE_VPI2
