#pragma once
#include <boost/core/noncopyable.hpp>
#include <semaphore.h>
#include <string.h>

class DepthMapSHM : public boost::noncopyable {
public:
  // Note that the constructor is only called when _creating_ the segment, not when opening it.
  // (only called on the producer)
  DepthMapSHM(size_t segmentSize) : m_segmentSize(segmentSize) {
    m_dataOffset = (sizeof(DepthMapSHM) + 4095) & (~4095);
    sem_init(&m_workerReadySem, /*pshared=*/ 1, /*value=*/ 0);
    sem_init(&m_workAvailableSem, /*pshared=*/ 1, /*value=*/ 0);

    // Initialize the 'finished' semaphore to 1, so we won't block waiting to read back a result that doesn't exist when we first start up the pipeline.
    sem_init(&m_workFinishedSem, /*pshared=*/ 1, /*value=*/ 1);
  }

  struct ViewParams {
    ViewParams() { memset(this, 0, sizeof(ViewParams)); }

    uint32_t width, height;
    uint32_t inputPitchBytes;
    uint32_t outputPitchBytes;

    // Offsets relative to segment data start
    uint32_t inputOffset[2]; // {left, right}
    uint32_t outputOffset;
  };

  char* data() { return reinterpret_cast<char*>(this) + m_dataOffset; }

  size_t m_segmentSize = 0;
  size_t m_dataOffset = 0;
  sem_t m_workerReadySem;
  sem_t m_workAvailableSem;
  sem_t m_workFinishedSem;


  uint32_t m_settingsGeneration = 0;

  static const size_t maxViews = 8;
  uint32_t m_activeViewCount = 0;
  ViewParams m_viewParams[maxViews];


  // === DGPU backend ===
  int m_algorithm = 2; // StereoSGM
  int m_numDisparities = 128;

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

  // === DepthAI backend ===
  int m_subpixelFractionalBits = 5; // 3...5.
  int m_confidenceThreshold = 245; // 0...255. Higher values allow lower-confidence samples through the filter.
  int m_medianFilter = 0; // valid: {0=disabled, 3=3x3 kernel, 5=5x5 kernel, 7=7x7 kernel}. Only enabled if subpixelFractionalBits == 3
  int m_bilateralFilterSigma = 0; // 0...65535. "larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color."
  int m_leftRightCheckThreshold = 10; // 0...255. only used if LR check is enabled. "Defines the maximum difference between the confidence of pixels from left-right and right-left confidence maps."
  bool m_enableLRCheck = false;
  bool m_enableSpatialFilter = false;
  bool m_enableTemporalFilter = false;

  // Profiling data
  float m_frameTimeMs = 0;
};

