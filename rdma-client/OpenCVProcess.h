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
#include "Matrices.h"
#include "Vectors.h"

class CameraSystem;
class RDMACameraProvider;


class OpenCVProcess {
public:
	OpenCVProcess(CameraSystem*, RDMACameraProvider*, size_t viewIdx);
	~OpenCVProcess();
	bool OpenCVAppStart();
	void OpenCVAppUpdate();
	void Thread();
	void Prerender();
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

  std::vector<float> m_geoDepthMapPositions; // CPU staging for m_geoDepthMapPositionBuffer
  RHIBuffer::ptr m_geoDepthMapPositionBuffer;
  RHIBuffer::ptr m_geoDepthMapTexcoordBuffer;
  RHIBuffer::ptr m_geoDepthMapTristripIndexBuffer;
  RHIBuffer::ptr m_geoDepthMapLineIndexBuffer;
  size_t m_geoDepthMapTristripIndexCount, m_geoDepthMapLineIndexCount;


  int m_iNextStereoAlgorithm;

  

	int m_iHasFrameForUpdate;
	int m_iDoneFrameOutput;

	// vr::CameraVideoStreamFrameHeader_t m_lastFrameHeader;
	Matrix4 m_lastFrameHeaderMatrix;
	float m_CameraDistanceMeters;
	std::thread * m_pthread;
	bool m_bQuitThread;
	float fNAN;

	bool m_bScreenshotNext;

  // Algorithm settings. Only committed on m_didChangeSettings = true.
  bool m_didChangeSettings;
  int m_algorithm;
  int m_blockSize;
  int m_preFilterCap;
  int m_uniquenessRatio;
  int m_speckleWindowSize;
  int m_speckleRange;


  bool m_useDepthBlur;

	//Matrices used in the stereo computation.
	cv::cuda::GpuMat origLeft_gpu;
	cv::cuda::GpuMat origRight_gpu;
	cv::Mat rectLeft;
	cv::cuda::GpuMat rectLeft_gpu;
	cv::cuda::GpuMat rectRight_gpu;
	cv::cuda::GpuMat resizedLeft_gpu;
	cv::cuda::GpuMat resizedRight_gpu;
	cv::cuda::GpuMat resizedLeftGray_gpu;
	cv::cuda::GpuMat resizedRightGray_gpu;
	cv::cuda::GpuMat resizedEqualizedLeftGray_gpu;
	cv::cuda::GpuMat resizedEqualizedRightGray_gpu;


	cv::Mat resizedLeftGray;
	cv::Mat resizedRightGray;
	cv::Mat mdisparity;
	cv::Mat mdisparity_expanded;
};
