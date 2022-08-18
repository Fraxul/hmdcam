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

  struct ViewDataVPI : public ViewData {
    ViewDataVPI() {
      CUDA_CHECK(cuStreamCreate(&m_cuStream, CU_STREAM_NON_BLOCKING));
      VPI_CHECK(vpiStreamCreateWrapperCUDA(m_cuStream, /*flags=*/ 0, &m_stream));
      VPI_CHECK(vpiEventCreate(/*flags=*/ 0, &m_frameStartedEvent));
      VPI_CHECK(vpiEventCreate(/*flags=*/ 0, &m_setupFinishedEvent));
      VPI_CHECK(vpiEventCreate(/*flags=*/ 0, &m_frameFinishedEvent));

      // Set up valid initial state for events
      vpiEventRecord(m_frameStartedEvent, m_stream);
      vpiEventRecord(m_setupFinishedEvent, m_stream);
      vpiEventRecord(m_frameFinishedEvent, m_stream);

      // Init NPP stream context
      memset(&m_nppStreamContext, 0, sizeof(m_nppStreamContext));
      nppSetStream(m_cuStream);
      nppGetStreamContext(&m_nppStreamContext);
    }

    virtual ~ViewDataVPI() {
      releaseVPIResources();

      vpiStreamDestroy(m_stream); m_stream = NULL;
      vpiEventDestroy(m_frameStartedEvent); m_frameStartedEvent = NULL;
      vpiEventDestroy(m_setupFinishedEvent); m_setupFinishedEvent = NULL;
      vpiEventDestroy(m_frameFinishedEvent); m_frameFinishedEvent = NULL;
      cuStreamDestroy(m_cuStream);
    }

    void releaseVPIResources() {
      PER_EYE({
        vpiPayloadDestroy(m_remapPayload[eyeIdx]);    m_remapPayload[eyeIdx] = NULL;
        vpiImageDestroy(m_grey[eyeIdx]);              m_grey[eyeIdx] = NULL;
        vpiImageDestroy(m_rectifiedGrey[eyeIdx]);     m_rectifiedGrey[eyeIdx] = NULL;
        vpiImageDestroy(m_resized[eyeIdx]);           m_resized[eyeIdx] = NULL;
        vpiImageDestroy(m_resizedTransposed[eyeIdx]); m_resizedTransposed[eyeIdx] = NULL;
      });

      vpiPayloadDestroy(m_disparityEstimator); m_disparityEstimator = NULL;

      vpiImageDestroy(m_disparity); m_disparity = NULL;
      vpiImageDestroy(m_confidence); m_confidence = NULL;

      vpiImageDestroy(m_disparityTransposed); m_disparityTransposed = NULL;
      vpiImageDestroy(m_confidenceTransposed); m_confidenceTransposed = NULL;
    }

    CUstream m_cuStream = 0;
    VPIStream m_stream = nullptr;
    NppStreamContext m_nppStreamContext;
    VPIEvent m_frameStartedEvent = nullptr;
    VPIEvent m_setupFinishedEvent = nullptr;
    VPIEvent m_frameFinishedEvent = nullptr;

    float m_setupTimeMs = 0, m_stereoTimeMs = 0;

    // Remap payloads for rectification
    VPIPayload m_remapPayload[2] = {nullptr, nullptr};

    // Output of format conversion from ArgusCamera shared images (NV12_ER)
    VPIImage m_grey[2] = {nullptr, nullptr};

    // Output of m_grey remap
    VPIImage m_rectifiedGrey[2] = {nullptr, nullptr};

    // Output of m_rectifiedGrey resize
    VPIImage m_resized[2] = {nullptr, nullptr};

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
