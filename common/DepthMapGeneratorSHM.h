#pragma once
#include "common/DepthMapGenerator.h"

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

  bool m_didChangeSettings = true; // force initial algorithm setup
  int m_disparityBytesPerPixel = 1;

  // DGPU backend -- initial algorithm settings
  int m_algorithm = 2;

  bool m_useDisparityFilter = true;
  int m_disparityFilterRadius = 3;
  int m_disparityFilterIterations = 1;

  // cuda::StereoBM, algorithm 0
  int m_sbmBlockSize = 19; // must be odd

  // cuda::StereoConstantSpaceBP, algorithm 1
  int m_sbpIterations = 5;
  int m_sbpLevels = 5;
  int m_scsbpNrPlane = 4;

  // cuda::StereoSGM, algorithm 2
  int m_sgmP1 = 10;
  int m_sgmP2 = 120;
  int m_sgmUniquenessRatio = 5; // 5-15
  bool m_sgmUseHH4 = true;

  // DepthAI worker data
  int m_confidenceThreshold = 230; // 0...255. Higher values allow lower-confidence samples through the filter.
  int m_medianFilter = 5; // valid: {0=disabled, 3=3x3 kernel, 5=5x5 kernel, 7=7x7 kernel}
  int m_bilateralFilterSigma = 0; // 0...65535. "larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color."
  int m_leftRightCheckThreshold = 4; // 0...255. only used if LR check is enabled. "Defines the maximum difference between the confidence of pixels from left-right and right-left confidence maps."
  bool m_enableLRCheck = false;


  struct ViewDataSHM : public ViewData {
    ViewDataSHM() {}

    size_t m_shmViewIndex;

    cv::cuda::GpuMat m_leftMap1_gpu, m_leftMap2_gpu, m_rightMap1_gpu, m_rightMap2_gpu;

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


  // Profiling events and data
  bool m_enableProfiling = false;
  cv::cuda::Event m_setupStartEvent;
  cv::cuda::Event m_setupFinishedEvent;

  cv::cuda::Event m_copyStartEvent;
  cv::cuda::Event m_processingFinishedEvent;
  cv::cuda::Stream m_globalStream;

  float m_setupTimeMs, m_syncTimeMs, m_algoTimeMs, m_copyTimeMs;

private:
  ViewDataSHM* viewDataAtIndex(size_t index) { return static_cast<ViewDataSHM*>(m_viewData[index]); }
};

