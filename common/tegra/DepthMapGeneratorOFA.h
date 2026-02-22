#pragma once
#include "common/DepthMapGenerator.h"
#include "common/tegra/NvSciCudaInterop.h"
#include "common/tegra/NvSciUtil.h"
#include "nvscibuf.h"
#include "nvscisync.h"
#include "nvmedia_iofa.h"
#include "rhi/cuda/CudaUtil.h"
#include <cuda.h>
#include <nppcore.h>

#define PER_EYE for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)

class DepthMapGeneratorOFA : public DepthMapGenerator {
public:
  DepthMapGeneratorOFA();
  virtual ~DepthMapGeneratorOFA();

protected:
  virtual void internalLoadSettings(cv::FileStorage&);
  virtual void internalSaveSettings(cv::FileStorage&);
  virtual void internalPostInitWithCameraSystem();
  virtual void internalProcessFrame();
  virtual void internalRenderIMGUI();
  virtual void internalRenderIMGUIPerformanceGraphs();

  struct ViewDataOFA : public ViewData {
    ViewDataOFA() {

    }

    void cleanup(NvSciCudaInteropBuffer*& buf) {
      if (!buf)
        return;
      assert(m_iofa); 
      NVMEDIA_CHECK(NvMediaIOFAUnregisterNvSciBufObj(m_iofa, buf->m_nvSciBuf));
      buf = nullptr;
    }

    void cleanup(NvSciCudaInteropSync*& sync) {
      if (!sync)
        return;
      assert(m_iofa); 
      NVMEDIA_CHECK(NvMediaIOFAUnregisterNvSciSyncObj(m_iofa, sync->m_nvSciSync));
      sync = nullptr;
    }

    virtual ~ViewDataOFA() {
      releaseResources();
    }

    void releaseResources() {
      cleanup(m_ofaPreSync);
      cleanup(m_ofaEofSync);
      cleanup(m_ofaInputBuffer[0]);
      cleanup(m_ofaInputBuffer[1]);
      cleanup(m_ofaOutputDisparityBuffer);
      cleanup(m_ofaOutputCostBuffer);
    }

    // Remap payloads for rectification
    cv::cuda::GpuMat m_undistortRectifyMap_gpu[2];

    // Output from remap+downsample
    cv::cuda::GpuMat m_rectifiedMat[2];

    // Reference to the IOFA, which is required to unregister buffers during destruction
    NvMediaIofa* m_iofa = nullptr;

    // OFA syncs
    NvSciCudaInteropSync* m_ofaPreSync = nullptr;
    NvSciCudaInteropSync* m_ofaEofSync = nullptr;

    // OFA input buffers
    NvSciCudaInteropBuffer* m_ofaInputBuffer[2] = {nullptr, nullptr};

    // OFA output buffers
    NvSciCudaInteropBuffer* m_ofaOutputDisparityBuffer = nullptr;
    NvSciCudaInteropBuffer* m_ofaOutputCostBuffer = nullptr;

    // Hold on to the NvMediaIofaBufArray to save a little CPU during frame processing,
    // since it doesn't change.
    NvMediaIofaBufArray m_ofaSurfArray;

    bool m_ofaSubmissionOK = false;
  };

  virtual ViewData* newEmptyViewData() { return new ViewDataOFA(); }
  virtual void internalUpdateViewData();

  CUevent m_masterFrameStartEvent;
  CUevent m_ofaHandoffCompleteEvent;
  CUevent m_masterFrameFinishedEvent;
  NvMediaIofa* m_iofa = nullptr;
  NvMediaIofaProcessParams m_iofaProcessParams;

  NvSciBufAttrList m_inputBufferAttrList = nullptr;
  NvSciBufAttrList m_disparityBufferAttrList = nullptr;
  NvSciBufAttrList m_costBufferAttrList = nullptr;


  float m_preOfaFrameTimeMs = 0;
  float m_ofaFrameTimeMs = 0;

  bool m_enableProfiling = true;

private:
  ViewDataOFA* viewDataAtIndex(size_t index) { return static_cast<ViewDataOFA*>(m_viewData[index]); }
};

