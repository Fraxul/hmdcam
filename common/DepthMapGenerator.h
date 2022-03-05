#pragma once
#include <glm/glm.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/affine.hpp"
#include <opencv2/core/cuda.hpp>
#include "opencv2/calib3d.hpp"
#include "rhi/RHISurface.h"
#include "rhi/RHIBuffer.h"
#include <vector>
#include "common/FxRenderView.h"
#include "common/DepthMapSHM.h"
#include "common/SHMSegment.h"

class CameraSystem;

class DepthMapGenerator {
public:
  DepthMapGenerator(CameraSystem*, SHMSegment<DepthMapSHM>*);
  ~DepthMapGenerator();

  void processFrame();
  void renderDisparityDepthMapStereo(size_t viewIdx, const FxRenderView& leftRenderView, const FxRenderView& rightRenderView, const glm::mat4& modelMatrix = glm::mat4(1.0f));
  void renderDisparityDepthMap(size_t viewIdx, const FxRenderView& renderView, const glm::mat4& modelMatrix = glm::mat4(1.0f));
  void renderIMGUI();

  RHISurface::ptr disparitySurface(size_t viewIdx) const { return m_viewData[viewIdx].m_disparityTexture; }
  RHISurface::ptr leftGrayscale(size_t viewIdx) const { return m_viewData[viewIdx].m_leftGray; }
  RHISurface::ptr rightGrayscale(size_t viewIdx) const { return m_viewData[viewIdx].m_rightGray; }

  // controls availability of leftGrayscale and rightGrayscale
  void setPopulateDebugTextures(bool value) { m_populateDebugTextures = value; }

  float m_disparityPrescale;


  bool loadSettings();
  void saveSettings();

protected:

  CameraSystem* m_cameraSystem;

  uint32_t  m_iFBSideWidth;
  uint32_t  m_iFBSideHeight;

  uint32_t  m_iFBAlgoWidth;
  uint32_t  m_iFBAlgoHeight;

  RHIBuffer::ptr m_geoDepthMapTexcoordBuffer;
  RHIBuffer::ptr m_geoDepthMapTristripIndexBuffer;
  RHIBuffer::ptr m_geoDepthMapLineIndexBuffer;
  size_t m_geoDepthMapTristripIndexCount, m_geoDepthMapLineIndexCount;

  // Depth map backend
  SHMSegment<DepthMapSHM>* m_depthMapSHM;

  // Algorithm settings
  bool m_didChangeSettings;
  int m_algorithm;
  int m_disparityBytesPerPixel;

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

  // DepthAI worker data
  int m_confidenceThreshold; // 0...255. Higher values allow lower-confidence samples through the filter.
  int m_medianFilter; // valid: {0=disabled, 3=3x3 kernel, 5=5x5 kernel, 7=7x7 kernel}
  int m_bilateralFilterSigma; // 0...65535. "larger value of the parameter means that farther colors within the pixel neighborhood will be mixed together, resulting in larger areas of semi-equal color."
  int m_leftRightCheckThreshold; // 0...255. only used if LR check is enabled. "Defines the maximum difference between the confidence of pixels from left-right and right-left confidence maps."
  bool m_enableLRCheck;


  // Render settings
  int m_trimLeft, m_trimTop;
  int m_trimRight, m_trimBottom;

  struct ViewData {
    ViewData() : m_isStereoView(false), m_isVerticalStereo(false) {}
    bool m_isStereoView;
    bool m_isVerticalStereo;
    size_t m_shmViewIndex;
    size_t m_leftCameraIndex, m_rightCameraIndex;

    cv::cuda::GpuMat m_leftMap1_gpu, m_leftMap2_gpu, m_rightMap1_gpu, m_rightMap2_gpu;
    glm::mat4 m_R1inv, m_Q, m_Qinv;

    RHISurface::ptr m_disparityTexture;
    RHISurface::ptr m_leftGray, m_rightGray;

    std::vector<RHIRenderTarget::ptr> m_disparityTextureMipTargets;

    // vr::CameraVideoStreamFrameHeader_t m_lastFrameHeader;
    // Matrix4 m_lastFrameHeaderMatrix;
    float m_CameraDistanceMeters;

    //Matrices used in the stereo computation.
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

  std::vector<ViewData> m_viewData;
  unsigned int m_viewDataRevision;

  void recomputePerViewData();

  // Profiling events and data
  bool m_enableProfiling;
  cv::cuda::Event m_setupStartEvent;
  cv::cuda::Event m_setupFinishedEvent;

  cv::cuda::Event m_copyStartEvent;
  cv::cuda::Event m_processingFinishedEvent;
  cv::cuda::Stream m_globalStream;

  float m_setupTimeMs, m_syncTimeMs, m_algoTimeMs, m_copyTimeMs;

  bool m_populateDebugTextures;

  void internalRenderSetup(size_t viewIdx, bool stereo, const FxRenderView& renderView0, const FxRenderView& renderView1, const glm::mat4& modelMatrix);
};
