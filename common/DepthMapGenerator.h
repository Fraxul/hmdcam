#pragma once
#include <glm/glm.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/affine.hpp"
#include <opencv2/core/cuda.hpp>
#include "opencv2/calib3d.hpp"
#include "rhi/RHISurface.h"
#include "rhi/RHIBuffer.h"
#include <thread>
#include "common/FxRenderView.h"
#include "common/DepthMapSHM.h"
#include "common/SHMSegment.h"

class CameraSystem;

class DepthMapGenerator {
public:
  DepthMapGenerator(CameraSystem*, SHMSegment<DepthMapSHM>*, size_t viewIdx);
  ~DepthMapGenerator();

  void processFrame();
  void renderDisparityDepthMap(const FxRenderView& renderView);
  void renderIMGUI();

  RHISurface::ptr disparitySurface() const { return m_disparityTexture; }
  RHISurface::ptr leftGrayscale() const { return m_leftGray; }
  RHISurface::ptr rightGrayscale() const { return m_rightGray; }

  // controls availability of leftGrayscale and rightGrayscale
  void setPopulateDebugTextures(bool value) { m_populateDebugTextures = value; }

  float m_disparityPrescale;

protected:
  cv::cuda::GpuMat m_leftMap1_gpu, m_leftMap2_gpu, m_rightMap1_gpu, m_rightMap2_gpu;
  glm::mat4 m_R1, m_R1inv, m_Q, m_Qinv;

  CameraSystem* m_cameraSystem;
  size_t m_viewIdx;

  uint32_t  m_iFBSideWidth;
  uint32_t  m_iFBSideHeight;

  uint32_t  m_iFBAlgoWidth;
  uint32_t  m_iFBAlgoHeight;

  RHISurface::ptr m_iTexture;
  RHISurface::ptr m_disparityTexture;
  RHISurface::ptr m_leftGray, m_rightGray;

  RHIBuffer::ptr m_geoDepthMapTexcoordBuffer;
  RHIBuffer::ptr m_geoDepthMapTristripIndexBuffer;
  RHIBuffer::ptr m_geoDepthMapLineIndexBuffer;
  size_t m_geoDepthMapTristripIndexCount, m_geoDepthMapLineIndexCount;


  // vr::CameraVideoStreamFrameHeader_t m_lastFrameHeader;
  // Matrix4 m_lastFrameHeaderMatrix;
  float m_CameraDistanceMeters;

  // Depth map backend
  SHMSegment<DepthMapSHM>* m_depthMapSHM;

  // Algorithm settings
  bool m_didChangeSettings;
  int m_algorithm;
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


  //Matrices used in the stereo computation.
  RHISurface::ptr origLeftBlitSrf;
  RHISurface::ptr origRightBlitSrf;
  RHIRenderTarget::ptr origLeftBlitRT;
  RHIRenderTarget::ptr origRightBlitRT;

  cv::cuda::GpuMat origLeft_gpu;
  cv::cuda::GpuMat origRight_gpu;
  cv::cuda::GpuMat rectLeft_gpu;
  cv::cuda::GpuMat rectRight_gpu;
  cv::cuda::GpuMat resizedLeft_gpu;
  cv::cuda::GpuMat resizedRight_gpu;
  cv::cuda::GpuMat resizedLeftGray_gpu;
  cv::cuda::GpuMat resizedRightGray_gpu;

  cv::cuda::Stream m_leftStream, m_rightStream;
  cv::cuda::Event m_leftRightJoinEvent;

  // Profiling events and data
  bool m_enableProfiling;
  cv::cuda::Event m_setupStartEvent;
  cv::cuda::Event m_setupFinishedEvent;

  cv::cuda::Event m_copyStartEvent;
  cv::cuda::Event m_processingFinishedEvent;

  float m_setupTimeMs, m_syncTimeMs, m_algoTimeMs, m_copyTimeMs;

  bool m_populateDebugTextures;

};