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

  void cleanup(NvSciCudaInteropBuffer*& buf) {
    if (!buf)
      return;
    assert(m_iofa);
    NVMEDIA_CHECK(NvMediaIOFAUnregisterNvSciBufObj(m_iofa, buf->m_nvSciBuf));
    delete buf;
    buf = nullptr;
  }

  void cleanup(NvSciCudaInteropSync*& sync) {
    if (!sync)
      return;
    assert(m_iofa);
    NVMEDIA_CHECK(NvMediaIOFAUnregisterNvSciSyncObj(m_iofa, sync->m_nvSciSync));
    delete sync;
    sync = nullptr;
  }


  struct SubmissionRingEntry {
    // OFA syncs
    NvSciCudaInteropSync* m_ofaPreSync = nullptr;
    NvSciCudaInteropSync* m_ofaEofSync = nullptr;

    CUevent m_ofaEofSyncDoneEvent = {0};

    // OFA input buffers
    NvSciCudaInteropBuffer* m_ofaInputBuffer[2] = {nullptr, nullptr};

    // OFA output buffers
    NvSciCudaInteropBuffer* m_ofaOutputDisparityBuffer = nullptr;
    NvSciCudaInteropBuffer* m_ofaOutputCostBuffer = nullptr;

    // Hold on to the NvMediaIofaBufArray to save a little CPU during frame processing,
    // since it doesn't change.
    NvMediaIofaBufArray m_ofaSurfArray;

    // True if this buffer-set is part of an active OFA submission.
    bool m_submissionActive = false;

    // True if the CPU has waited for the EOF fence to ensure that it's actually clear of the OFA.
    // Set to false when submitting this entry to OFA. Must clear before resubmit.
    bool m_fenceClearedOnCPU = true;
  };

  std::array<SubmissionRingEntry*, 8> m_ring;
  uint32_t m_nextRingIndex = 0;

  struct ViewDataOFA : public ViewData {
    ViewDataOFA() {
    }

    virtual ~ViewDataOFA() {
    }

    // Remap payloads for rectification
    cv::cuda::GpuMat m_undistortRectifyMap_gpu[2];

    // The remapped luma is kept on the base class as m_rectifiedLuma[2].

    // Which buffer in the ring was last submitted by this view.
    // -1 means there's no submission pending.
    int32_t m_activeRingIndex = -1;
  };

  virtual ViewData* newEmptyViewData() { return new ViewDataOFA(); }
  virtual void internalUpdateViewData();

  CUevent m_masterFrameStartEvent;
  CUevent m_ofaHandoffCompleteEvent;
  CUevent m_masterFrameFinishedEvent;
  NvMediaIofa* m_iofa = nullptr;
  NvMediaIofaProcessParams m_iofaProcessParams;
  NvSciSyncCpuWaitContext m_cpuWaitContext;

  NvSciBufAttrList m_inputBufferAttrList = nullptr;
  NvSciBufAttrList m_disparityBufferAttrList = nullptr;
  NvSciBufAttrList m_costBufferAttrList = nullptr;

  float m_preOfaFrameTimeMs = 0;
  float m_ofaFrameTimeMs = 0;

  bool m_enableProfiling = true;

  // Cost-to-confidence mapping
  uint8_t m_lowCostThreshold = 4;
  uint8_t m_highCostThreshold = 48;
  float m_costCurve = 1.0f;


private:
  ViewDataOFA* viewDataAtIndex(size_t index) { return static_cast<ViewDataOFA*>(m_viewData[index]); }
};
