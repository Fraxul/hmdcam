#pragma once
#include "common/DepthMapGenerator.h"
#include "common/ScrollingBuffer.h"

class DepthMapGeneratorSHM : public DepthMapGenerator {
public:
  DepthMapGeneratorSHM(DepthMapGeneratorBackend);
  virtual ~DepthMapGeneratorSHM();

  SHMSegment<DepthMapSHM>* depthMapSHM() const { return m_depthMapSHM; }

protected:
  SHMSegment<DepthMapSHM>* m_depthMapSHM;

  int spawnDepthWorker();
  void waitForDepthWorkerReady(int pid, sem_t* sem, unsigned int timeout_sec);

  virtual void internalLoadSettings(cv::FileStorage&);
  virtual void internalSaveSettings(cv::FileStorage&);
  virtual void internalProcessFrame();
  virtual void internalRenderIMGUI();
  virtual void internalRenderIMGUIPerformanceGraphs();

  bool m_didChangeSettings = true; // force initial algorithm setup
  int m_disparityBytesPerPixel = 1;

  struct ViewDataSHM : public ViewData {
    ViewDataSHM() {}
    virtual ~ViewDataSHM() {}

    size_t m_shmViewIndex;

    cv::cuda::GpuMat m_undistortRectifyMap_gpu[2];

    RHISurface::ptr origLeftBlitSrf;
    RHISurface::ptr origRightBlitSrf;
    RHIRenderTarget::ptr origLeftBlitRT;
    RHIRenderTarget::ptr origRightBlitRT;

    cv::cuda::GpuMat rectLeft_gpu;
    cv::cuda::GpuMat rectRight_gpu;
    cv::cuda::GpuMat resizedLeft_gpu;
    cv::cuda::GpuMat resizedRight_gpu;
    cv::cuda::GpuMat resizedTransposedLeft_gpu;
    cv::cuda::GpuMat resizedTransposedRight_gpu;
  };
  virtual ViewData* newEmptyViewData() { return new ViewDataSHM(); }
  virtual void internalUpdateViewData();

  struct ProfilingData {
    float m_setupTimeMs = 0;
    float m_syncTimeMs = 0;
    float m_algoTimeMs = 0;
    float m_copyTimeMs = 0;
    float m_processingTimedOutThisFrame = 0;
  };
  ScrollingBuffer<ProfilingData> m_profilingDataBuffer = ScrollingBuffer<ProfilingData>(128);

  ProfilingData m_profilingData;


  // Profiling events and data
  bool m_enableProfiling = true;
  bool m_haveValidProfilingData = false;
  cv::cuda::Event m_setupStartEvent;
  cv::cuda::Event m_setupFinishedEvent;

  cv::cuda::Event m_copyStartEvent;
  cv::cuda::Event m_processingFinishedEvent;
  cv::cuda::Stream m_globalStream;

private:
  ViewDataSHM* viewDataAtIndex(size_t index) { return static_cast<ViewDataSHM*>(m_viewData[index]); }
};

