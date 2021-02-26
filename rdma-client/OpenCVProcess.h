#pragma once
#include <glm/glm.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/affine.hpp"
#include <opencv2/core/cuda.hpp>
#include "opencv2/calib3d.hpp"
#include <opencv2/cudastereo.hpp>
#include "rhi/RHISurface.h"
#include "rhi/RHIBuffer.h"
#include <thread>
#include "Matrices.h"
#include "Vectors.h"
#include "FxRenderView.h"

class CameraSystem;
class RDMACameraProvider;


class OpenCVProcess {
public:
  OpenCVProcess(CameraSystem*, RDMACameraProvider*, size_t viewIdx);
  ~OpenCVProcess();
  bool OpenCVAppStart();
  void OpenCVAppUpdate();
  void DrawDisparityDepthMap(const FxRenderView& renderView);
  void DrawUI();
  void TakeScreenshot();

  void ConvertToGray( cv::InputArray src, cv::OutputArray dst );
  void BlurDepths();
  Vector4 TransformToWorldSpace( float x, float y, int disp );
  Vector4 TransformToLocalSpace( float x, float y, int disp );

  // vr::TrackedCameraHandle_t m_pCamera;

  // vr::HmdMatrix34_t  m_headFromCamera[2];
  cv::cuda::GpuMat m_leftMap1_gpu, m_leftMap2_gpu, m_rightMap1_gpu, m_rightMap2_gpu;
  Matrix4 m_R1, m_R1inv, m_Q, m_Qinv;
  cv::Ptr< cv::StereoMatcher > m_stereo;
  cv::Ptr<cv::cuda::DisparityBilateralFilter> m_disparityFilter;

  CameraSystem* m_cameraSystem;
  size_t m_viewIdx;
  RDMACameraProvider* m_cameraProvider;

  uint32_t  m_iFBSideWidth;
  uint32_t  m_iFBSideHeight;
  std::vector< float > m_valids;
  std::vector< float > m_depths;

  uint32_t  m_iFBAlgoWidth;
  uint32_t  m_iFBAlgoHeight;
  uint32_t  m_iProcFrames;
  uint32_t  m_iFramesSinceFPS, m_iFPS;
  double    m_dTimeOfLastFPS;

  RHISurface::ptr m_iTexture;
  RHISurface::ptr m_disparityTexture;
  RHISurface::ptr m_leftGray, m_rightGray;

  RHIBuffer::ptr m_geoDepthMapTexcoordBuffer;
  RHIBuffer::ptr m_geoDepthMapTristripIndexBuffer;
  RHIBuffer::ptr m_geoDepthMapLineIndexBuffer;
  size_t m_geoDepthMapTristripIndexCount, m_geoDepthMapLineIndexCount;


  int m_iHasFrameForUpdate;
  int m_iDoneFrameOutput;

  // vr::CameraVideoStreamFrameHeader_t m_lastFrameHeader;
  // Matrix4 m_lastFrameHeaderMatrix;
  float m_CameraDistanceMeters;
  float fNAN;

  // Algorithm settings. Only committed on m_didChangeSettings = true.
  bool m_didChangeSettings;
  int m_algorithm;
  bool m_useDisparityFilter;
  int m_disparityFilterRadius;
  int m_disparityFilterIterations;

  // cuda::StereoBM
  int m_sbmBlockSize;

  // cuda::StereoBeliefPropagation
  int m_sbpIterations;
  int m_sbpLevels;

  // cuda::StereoConstantSpaceBP (shares parameters with StereoBeliefPropagation)
  int m_scsbpNrPlane;

  // cuda::StereoSGM
  int m_sgmP1;
  int m_sgmP2;
  int m_sgmUniquenessRatio;

  bool m_useDepthBlur;

  //Matrices used in the stereo computation.
  cv::cuda::GpuMat origLeft_gpu;
  cv::cuda::GpuMat origRight_gpu;
  cv::cuda::GpuMat rectLeft_gpu;
  cv::cuda::GpuMat rectRight_gpu;
  cv::cuda::GpuMat resizedLeft_gpu;
  cv::cuda::GpuMat resizedRight_gpu;
  cv::cuda::GpuMat resizedLeftGray_gpu;
  cv::cuda::GpuMat resizedRightGray_gpu;
  cv::cuda::GpuMat resizedEqualizedLeftGray_gpu;
  cv::cuda::GpuMat resizedEqualizedRightGray_gpu;

  cv::cuda::GpuMat mdisparity_gpu;
  cv::cuda::GpuMat mdisparity_filtered_gpu;
  cv::Mat mdisparity_8uc1; // for StereoBM only
  cv::Mat mdisparity;

  cv::cuda::Stream m_leftStream, m_rightStream;
  cv::cuda::Event m_leftRightJoinEvent;
};
