#pragma once
#include <glm/glm.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/core/affine.hpp"
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
	cv::Mat m_leftMap1, m_leftMap2, m_rightMap1, m_rightMap2;
	Matrix4 m_R1, m_R1inv, m_Q, m_Qinv;
	cv::Ptr< cv::StereoSGBM > m_stereo;

	CameraSystem* m_cameraSystem;
  size_t m_viewIdx;
  RDMACameraProvider* m_cameraProvider;

	uint8_t * m_pFBSides[2];
	uint32_t * m_pFBSidesColor[2];
	uint16_t * m_pDisparity;
	uint32_t * m_pColorOut;
	uint32_t * m_pColorOut2;
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
	cv::Mat m_cvQ;
	std::thread * m_pthread;
	bool m_bQuitThread;
	float fNAN;

	bool m_bScreenshotNext;
	int m_iCurrentStereoAlgorithm;

	//Matrices used in the stereo computation.
	cv::Mat origLeft;
	cv::Mat origRight;
	cv::Mat rectLeft;
	cv::Mat rectRight;
	cv::Mat resizedLeftGray;
	cv::Mat resizedRightGray;
	cv::Mat resizedLeft;
	cv::Mat resizedRight;
	cv::Mat mdisparity;
	cv::Mat mdisparity_expanded;
};
