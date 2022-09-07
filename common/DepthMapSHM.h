#pragma once
#include <boost/core/noncopyable.hpp>
#include <semaphore.h>

class DepthMapSHM : public boost::noncopyable {
public:
  DepthMapSHM(size_t segmentSize) : m_segmentSize(segmentSize) {
    m_dataOffset = (sizeof(DepthMapSHM) + 4095) & (~4095);
    sem_init(&m_workerReadySem, /*pshared=*/ 1, /*value=*/ 0);
    sem_init(&m_workAvailableSem, /*pshared=*/ 1, /*value=*/ 0);

    // Initialize the 'finished' semaphore to 1, so we won't block waiting to read back a result that doesn't exist when we first start up the pipeline.
    sem_init(&m_workFinishedSem, /*pshared=*/ 1, /*value=*/ 1);
  }

  struct ViewParams {
    uint32_t width, height;
    uint32_t inputPitchBytes;
    uint32_t outputPitchBytes;

    // Offsets relative to segment data start
    uint32_t inputLeftOffset;
    uint32_t inputRightOffset;
    uint32_t outputOffset;
  };



  char* data() { return reinterpret_cast<char*>(this) + m_dataOffset; }

  size_t m_segmentSize;
  size_t m_dataOffset;
  sem_t m_workerReadySem;
  sem_t m_workAvailableSem;
  sem_t m_workFinishedSem;


  static const size_t maxViews = 8;
  uint32_t m_activeViewCount;
  ViewParams m_viewParams[maxViews];
  unsigned int m_settingsGeneration;

  // --- DGPU worker data ---
  // Algorithm settings
  int m_algorithm;
  int m_numDisparities;
  bool m_useDisparityFilter;
  int m_disparityFilterRadius;
  int m_disparityFilterIterations;

  // cuda::StereoBM
  int m_sbmBlockSize;

  // cuda::StereoConstantSpaceBP
  int m_sbpIterations;
  int m_sbpLevels;
  int m_scsbpNrPlane;

  // cuda::StereoSGM
  int m_sgmP1;
  int m_sgmP2;
  int m_sgmUniquenessRatio;
  bool m_sgmUseHH4;

  // --- DepthAI worker data ---

  int m_confidenceThreshold; // 0...255. Higher values allow lower-confidence samples through the filter.
  int m_medianFilter; // valid: {0=disabled, 3=3x3 kernel, 5=5x5 kernel, 7=7x7 kernel}
  int m_bilateralFilterSigma; // 0...65535. "larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color."
  int m_leftRightCheckThreshold; // 0...128. only used if LR check is enabled. "Defines the maximum difference between the confidence of pixels from left-right and right-left confidence maps."
  bool m_enableLRCheck;
  bool m_enableSpatialFilter;
  bool m_enableTemporalFilter;


  // Profiling data
  float m_frameTimeMs;
};

