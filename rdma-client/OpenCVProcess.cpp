#include "imgui.h"
#include "OpenCVProcess.h"
#include "common/CameraSystem.h"
#include "RDMACameraProvider.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/RHICVInterop.h"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudastereo.hpp>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <thread>
#include "Matrices.h"
#include "Vectors.h"

#include "stb/stb_image_write.h"


RHIRenderPipeline::ptr disparityDepthMapPipeline;
FxAtomicString ksMeshDisparityDepthMapUniformBlock("MeshDisparityDepthMapUniformBlock");
static FxAtomicString ksDisparityTex("disparityTex");
static FxAtomicString ksImageTex("imageTex");
struct MeshDisparityDepthMapUniformBlock {
  glm::mat4 modelViewProjection;
  glm::mat4 R1inv;
  float Q3, Q7, Q11;
  float CameraDistanceMeters;

  glm::vec2 mogrify;
  float pad3, pad4;
};

#define NUM_DISP 128 //Max disparity. Must be in {64, 128, 256} for CUDA algorithms.

inline double * CoPTr( const std::initializer_list<double>& d ) { return (double*)d.begin(); }
#define DO_FISHEYE 1

#define DO_PROFILE 1
#if DO_PROFILE
#define PROFILE( x ) { double Now = OGGetAbsoluteTime(); ImGui::TextUnformatted(x); ImGui::NextColumn(); ImGui::Text("%.3fms", (Now-Start)*1000.0); ImGui::NextColumn(); Start = Now; }
#else
#define PROFILE( x )
#endif

static void OGUSleep( int ius ) {
	usleep( ius );
}

static double OGGetAbsoluteTime() {
	struct timeval tv;
	gettimeofday( &tv, 0 );
	return ((double)tv.tv_usec)/1000000. + (tv.tv_sec);
}

Matrix4 Matrix4FromCVMatrix( cv::Mat matin )
{
	Matrix4 out;
	out.identity();
	for ( int y = 0; y < matin.rows; y++ )
	{
		for ( int x = 0; x < matin.cols; x++ )
		{
			out[x + y * 4] = (float)matin.at<double>( y, x );
		}
	}
	return out;
}

OpenCVProcess::OpenCVProcess(CameraSystem* cs, RDMACameraProvider* cp, size_t viewIdx) :
	  m_cameraSystem(cs)
  , m_viewIdx(viewIdx)
  , m_cameraProvider(cp)
	, m_iProcFrames( 0 )
	, m_iFramesSinceFPS( 0 )
	, m_dTimeOfLastFPS( 0 ) 
  , m_useDepthBlur(false)
  , m_leftRightJoinEvent(cv::cuda::Event::DISABLE_TIMING)
{
  if (!disparityDepthMapPipeline) {
    disparityDepthMapPipeline = rhi()->compileRenderPipeline("shaders/meshDisparityDepthMap.vtx.glsl", "shaders/meshTexture.frag.glsl", RHIVertexLayout({
        RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "textureCoordinates", 0, sizeof(float) * 4)
      }), kPrimitiveTopologyTriangleStrip);
  }

	fNAN = nanf( "" );


  m_didChangeSettings = false;
  m_algorithm = 2;
  m_useDisparityFilter = true;
  m_disparityFilterRadius = 3;
  m_disparityFilterIterations = 1;

  // cuda::StereoBM
  m_sbmBlockSize = 19; // must be odd

  // cuda::StereoBeliefPropagation
  m_sbpIterations = 5;
  m_sbpLevels = 5;

  // cuda::StereoConstantSpaceBP
  m_scsbpNrPlane = 4;

  // cuda::StereoSGM
  m_sgmP1 = 10;
  m_sgmP2 = 120;
  m_sgmUniquenessRatio = 5; // 5-15


}

OpenCVProcess::~OpenCVProcess()
{
}

bool OpenCVProcess::OpenCVAppStart()
{
#define MOGRIFY_X 4
#define MOGRIFY_Y 4
#define IGNORE_EDGE_DATA_PIXELS 16


  CameraSystem::View& v = m_cameraSystem->viewAtIndex(m_viewIdx);
  CameraSystem::Camera& cL = m_cameraSystem->cameraAtIndex(v.cameraIndices[0]);
  CameraSystem::Camera& cR = m_cameraSystem->cameraAtIndex(v.cameraIndices[1]);

  // TODO validate CameraSystem:::updateViewStereoDistortionParameters against the distortion map initialization code above

  // cv::initUndistortRectifyMap( K1, D1, R1, P1, cv::Size( 960, 960 ), CV_16SC2, m_leftMap1, m_leftMap2 );
  cv::Size imageSize = cv::Size(m_cameraProvider->streamWidth(), m_cameraProvider->streamHeight());
#if 1
  {
    cv::Mat m1, m2;
    cv::initUndistortRectifyMap(cL.intrinsicMatrix, cL.distCoeffs, v.stereoRectification[0], v.stereoProjection[0], imageSize, CV_32F, m1, m2);
    m_leftMap1_gpu.upload(m1); m_leftMap2_gpu.upload(m2);
    cv::initUndistortRectifyMap(cR.intrinsicMatrix, cR.distCoeffs, v.stereoRectification[1], v.stereoProjection[1], imageSize, CV_32F, m1, m2);
    m_rightMap1_gpu.upload(m1); m_rightMap2_gpu.upload(m2);
    
  }
	m_R1 = Matrix4FromCVMatrix( v.stereoRectification[0] );
	m_R1inv = m_R1;
	m_R1inv = m_R1inv.invert();
	m_Q = Matrix4FromCVMatrix( v.stereoDisparityToDepth );
#else
	cv::Mat R1, R2, P1, P2, cvQ;
  cv::stereoRectify( cL.intrinsicMatrix, cL.distCoeffs, cR.intrinsicMatrix, cR.distCoeffs, imageSize, v.stereoRotation, v.stereoTranslation, R1, R2, P1, P2, cvQ, cv::CALIB_ZERO_DISPARITY );
  cv::initUndistortRectifyMap(cL.intrinsicMatrix, cL.distCoeffs, v.stereoRectification[0], v.stereoProjection[0], imageSize, CV_16SC2, m_leftMap1, m_leftMap2);
  cv::initUndistortRectifyMap(cR.intrinsicMatrix, cR.distCoeffs, v.stereoRectification[1], v.stereoProjection[1], imageSize, CV_16SC2, m_rightMap1, m_rightMap2);
	m_R1 = Matrix4FromCVMatrix( R1 );
	m_R1inv = m_R1;
	m_R1inv = m_R1inv.invert();
	m_Q = Matrix4FromCVMatrix( cvQ );
#endif

	m_CameraDistanceMeters = glm::length(glm::vec3(v.stereoTranslation.at<double>(0), v.stereoTranslation.at<double>(1), v.stereoTranslation.at<double>(2)));


	m_iFBSideWidth = m_cameraProvider->streamWidth();
	m_iFBSideHeight = m_cameraProvider->streamHeight();

	m_iFBAlgoWidth = m_iFBSideWidth / MOGRIFY_X;
	m_iFBAlgoHeight = m_iFBSideHeight / MOGRIFY_Y;

  m_iTexture = rhi()->newTexture2D(m_iFBSideWidth, m_iFBSideHeight, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
  m_disparityTexture = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor(kSurfaceFormat_R16));
  m_leftGray = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));
  m_rightGray = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));


	m_valids.resize( m_iFBAlgoHeight  * m_iFBAlgoWidth );
	m_valids.assign( m_valids.size(), 1 );
	m_depths.resize( m_iFBAlgoHeight  * m_iFBAlgoWidth );

  //Set up what matrices we can to prevent dynamic memory allocation.

  origLeft_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);
  origRight_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);

  rectLeft_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);
  rectRight_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);

  resizedLeft_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth, CV_8UC4);
  resizedRight_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth, CV_8UC4);

  resizedLeftGray_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);
  resizedRightGray_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);

  resizedEqualizedLeftGray_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);
  resizedEqualizedRightGray_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);

	mdisparity = cv::Mat( m_iFBAlgoHeight, m_iFBAlgoWidth, CV_16S );
	mdisparity_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth, CV_16S );


  // Set up geometry depth map data

  // Set up geometry map texcoord and index buffers
  // (borrowed from camera_app.cpp)
	{
    { // Texcoord and position buffers
      std::vector<float> depth_tc;
      int uiDepthVertCount = m_iFBAlgoWidth * m_iFBAlgoHeight;
      depth_tc.resize( uiDepthVertCount * 4);
      for ( int y = 0; y < m_iFBAlgoHeight; y++ ) {
        for ( int x = 0; x < m_iFBAlgoWidth; x++ ) {
          depth_tc[(x + y * m_iFBAlgoWidth) * 4 + 0] = 1.0f * x / (m_iFBAlgoWidth - 1);
          depth_tc[(x + y * m_iFBAlgoWidth) * 4 + 1] = 1.0f * y / (m_iFBAlgoHeight - 1);
          depth_tc[(x + y * m_iFBAlgoWidth) * 4 + 2] = x;
          depth_tc[(x + y * m_iFBAlgoWidth) * 4 + 3] = y;
        }
      }
      m_geoDepthMapTexcoordBuffer = rhi()->newBufferWithContents(depth_tc.data(), depth_tc.size() * sizeof(float), kBufferUsageCPUWriteOnly);
    }


    int dmxm1 = m_iFBAlgoWidth - 1;
    int dmym1 = m_iFBAlgoHeight - 1;
    { // Tristrip indices
      //From https://github.com/cnlohr/spreadgine/blob/master/src/spreadgine_util.c:216
      std::vector<uint32_t> depth_ia;
      depth_ia.resize( m_iFBAlgoWidth * dmym1 * 2 );
      //int uiDepthIndexCount = (unsigned int)depth_ia.size();
      for ( int y = 0; y < dmym1; y++ )
      {
        for ( int x = 0; x < m_iFBAlgoWidth; x++ )
        {
          int sq = (x + y * dmxm1) * 2;
          depth_ia[sq + 0] = x + y * (m_iFBAlgoWidth);
          depth_ia[sq + 1] = (x)+(y + 1) * (m_iFBAlgoWidth);
        }
      }

      m_geoDepthMapTristripIndexBuffer = rhi()->newBufferWithContents(depth_ia.data(), depth_ia.size() * sizeof(uint32_t), kBufferUsageCPUWriteOnly);
      m_geoDepthMapTristripIndexCount = depth_ia.size();
    }

    { // Line indices
      std::vector<uint32_t> depth_ia_lines;
      depth_ia_lines.resize( m_iFBAlgoWidth * dmym1 * 2 );
      //int uiDepthIndexCountLines = (unsigned int)depth_ia_lines.size();

      for ( int y = 0; y < dmym1; y++ )
      {
        for ( int x = 0; x < m_iFBAlgoWidth; x += 2 )
        {
          int sq = (x + y * dmxm1) * 2;
          depth_ia_lines[sq + 0] = x + y * (m_iFBAlgoWidth);
          depth_ia_lines[sq + 1] = (x + 1) + (y) * (m_iFBAlgoWidth);
          depth_ia_lines[sq + 2] = (x + 1) + (y + 1) * (m_iFBAlgoWidth);
          depth_ia_lines[sq + 3] = (x + 2) + (y + 1) * (m_iFBAlgoWidth);
        }
      }
      m_geoDepthMapLineIndexBuffer = rhi()->newBufferWithContents(depth_ia_lines.data(), depth_ia_lines.size() * sizeof(uint32_t), kBufferUsageCPUWriteOnly);
      m_geoDepthMapLineIndexCount = depth_ia_lines.size();
    }
	}

  return true;
}

void OpenCVProcess::ConvertToGray( cv::InputArray src, cv::OutputArray dst )
{
	//You can do this the OpenCV way, but my alternative transform seems more successful.
	if ( 1 )
	{
	}
	else
	{
		cv::Mat msrc( src.getMat() );
		cv::Mat mdst( dst.getMat() );

		int w = msrc.cols;
		int h = msrc.rows;

		for (int y = 0; y < h; y++ ) {

			for (int x = 0; x < w; x++ ) {
				uint32_t inx = msrc.at<uint32_t>(y, x);

				int r = (inx >> 0) & 0xff;
				int g = (inx >> 8) & 0xff;
				int b = (inx >> 16) & 0xff;
        mdst.at<uint8_t>(y, x) = (uint8_t)((r + g + b) / 3);

			}
		}
	}
}

Vector4 OpenCVProcess::TransformToLocalSpace( float x, float y, int disp )
{
	float fDisp = ( float ) disp / 16.f; //  16-bit fixed-point disparity map (where each disparity value has 4 fractional bits)

	float lz = m_Q[11] * m_CameraDistanceMeters / ( fDisp * MOGRIFY_X );
	float ly = -(y * MOGRIFY_Y + m_Q[7]) / m_Q[11];
	float lx = (x * MOGRIFY_X + m_Q[3]) / m_Q[11];
	lx *= lz;
	ly *= lz;
	lz *= -1;
	return m_R1inv * Vector4( lx, ly, lz, 1.0 );
}

Vector4 OpenCVProcess::TransformToWorldSpace( float x, float y, int disp )
{
	Vector4 local = TransformToLocalSpace( x, y, disp );
#if 0
	Matrix4 mlefteye = m_lastFrameHeaderMatrix;
	Vector4 placerelview( local );
	Vector4 Worldview = mlefteye * (placerelview);
	return Worldview;
#else
  // No local-to-world transform required, since we don't have a global tracking matrix.
  return local;
#endif
}

#if 0
void OpenCVProcess::BlurDepths()
{
	//This does an actual blurring function. 
#define RIGHT_EDGE_NO_TRUST 24

	std::vector< float > validsback( m_iFBAlgoHeight  * m_iFBAlgoWidth );
	std::vector< float > depthsback( m_iFBAlgoHeight  * m_iFBAlgoWidth );

	//Initialize the data for this frame
	for ( unsigned y = 0; y < m_iFBAlgoHeight; y++ )
	{
		for ( unsigned x = 0; x < m_iFBAlgoWidth; x++ )
		{
      int16_t pxi = mdisparity.at<int16_t>(y,x);
			int idx = (y * m_iFBAlgoWidth) + x;
			if ( pxi <= 0 || pxi >= m_iFBAlgoWidth * 16 )
			{
				//If we don't know the depth, then we just discard it.  We could handle that here.  Additionally,
				//if we wanted, we could emit fake dots where we believe the floor to be.
				//Right now, we do nothing.
			}
			else
			{
				m_valids[idx] = 1.0;
				m_depths[idx] = pxi;
			}

			//Never trust the right side of the screen.
			if ( x >= m_iFBAlgoWidth - RIGHT_EDGE_NO_TRUST )
			{
				m_valids[idx] = 0;
				m_depths[idx] = 0;
			}
		}
	}
	for ( int iter = 0; iter < 10; iter++ )
	{
		for ( unsigned y = 4; y < m_iFBAlgoHeight - 4; y++ )
		{
			for ( unsigned x = 4; x < m_iFBAlgoWidth - 4; x++ )
			{
				int idx = (y * m_iFBAlgoWidth) + x;
				float tval = 0;
				float tdepths = 0;
				for ( int ly = -1; ly <= 1; ly++ )
					for ( int lx = -1; lx <= 1; lx++ )
					{
						int idxx = (y+ly) * m_iFBAlgoWidth + x+lx;
						tval += m_valids[idxx];
						tdepths += m_depths[idxx];
					}
				validsback[idx] = tval / 9;
				depthsback[idx] = tdepths / 9;
			}
		}
		memcpy( &m_valids[0], &validsback[0], sizeof( float ) * validsback.size() );
		memcpy( &m_depths[0], &depthsback[0], sizeof( float ) * depthsback.size() );

	}

	for ( unsigned y = 0; y < m_iFBAlgoHeight; y++ )
	{
		for ( unsigned x = 0; x < m_iFBAlgoWidth; x++ )
		{
			int idx = y * m_iFBAlgoWidth + x;
			if ( x < 1 || x >= m_iFBAlgoWidth - 1 )
			{
				//Must throw out edges.
        mdisparity.at<uint16_t>(y,x) = 0xfff0;
				continue;
			}
			uint16_t pxi = mdisparity.at<uint16_t>(y,x);
			if ( pxi == 0 || pxi >= m_iFBAlgoWidth * 16 )
			{
				if ( m_valids[idx] < .00005 )
				{
					m_valids[idx] = 0;
					m_depths[idx] = 0;
          mdisparity.at<uint16_t>(y,x) = 0xfff0;
				}
				else
				{
					mdisparity.at<uint16_t>(y,x) = (uint16_t)(m_depths[idx] / m_valids[idx]);
				}
			}
			m_valids[idx] *= .9f;
			m_depths[idx] *= .9f;
		}
	}
}
#endif

void OpenCVProcess::OpenCVAppUpdate()
{
#if DO_PROFILE
  ImGui::Begin("CV Profile");

  m_iProcFrames++;
  m_iFramesSinceFPS++;

	double Start = OGGetAbsoluteTime();
  if ( Start >= m_dTimeOfLastFPS + 1 )
  {
    if ( Start - m_dTimeOfLastFPS < 4 )
      m_dTimeOfLastFPS++;
    else
      m_dTimeOfLastFPS = Start;
    m_iFPS = m_iFramesSinceFPS;
    m_iFramesSinceFPS = 0;
  }
  ImGui::Text("Frames: %5d; %3d FPS", m_iProcFrames, m_iFPS );

  ImGui::Columns(2);
#endif


  if ((!m_stereo) || m_didChangeSettings) {
    m_sbmBlockSize |= 1; // enforce odd blockSize
    m_disparityFilterRadius |= 1; // enforce odd filter size

    switch (m_algorithm) {
      default:
        m_algorithm = 0;
      case 0:
        // uses CV_8UC1 disparity
        m_stereo = cv::cuda::createStereoBM(NUM_DISP, m_sbmBlockSize);
        break;
      case 1:
        m_stereo = cv::cuda::createStereoBeliefPropagation(NUM_DISP, m_sbpIterations, m_sbpLevels);
        break;
      case 2: {
        if (m_sbpLevels > 4) // more than 4 levels seems to trigger an internal crash
          m_sbpLevels = 4;
        m_stereo = cv::cuda::createStereoConstantSpaceBP(NUM_DISP, m_sbpIterations, m_sbpLevels, m_scsbpNrPlane);
      } break;
      case 3:
        m_stereo = cv::cuda::createStereoSGM(0, NUM_DISP, m_sgmP1, m_sgmP2, m_sgmUniquenessRatio, cv::cuda::StereoSGM::MODE_HH4);
        break;
    };

    if (m_useDisparityFilter) {
      m_disparityFilter = cv::cuda::createDisparityBilateralFilter(NUM_DISP, m_disparityFilterRadius, m_disparityFilterIterations);
    }
  }

	origLeft_gpu.upload(m_cameraProvider->cvMat(m_cameraSystem->viewAtIndex(m_viewIdx).cameraIndices[0]), m_leftStream);
	origRight_gpu.upload(m_cameraProvider->cvMat(m_cameraSystem->viewAtIndex(m_viewIdx).cameraIndices[1]), m_rightStream);

	cv::cuda::remap( origLeft_gpu, rectLeft_gpu, m_leftMap1_gpu, m_leftMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue=*/ cv::Scalar(), m_leftStream);
	cv::cuda::remap( origRight_gpu, rectRight_gpu, m_rightMap1_gpu, m_rightMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue=*/ cv::Scalar(), m_rightStream);

	cv::cuda::resize( rectLeft_gpu, resizedLeft_gpu, cv::Size( m_iFBAlgoWidth, m_iFBAlgoHeight ), 0, 0, cv::INTER_LINEAR, m_leftStream);
	cv::cuda::resize( rectRight_gpu, resizedRight_gpu, cv::Size( m_iFBAlgoWidth, m_iFBAlgoHeight ), 0, 0, cv::INTER_LINEAR, m_rightStream);

  cv::cuda::cvtColor( resizedLeft_gpu, resizedLeftGray_gpu, cv::COLOR_BGRA2GRAY, 0, m_leftStream);
  cv::cuda::cvtColor( resizedRight_gpu, resizedRightGray_gpu, cv::COLOR_BGRA2GRAY, 0, m_rightStream);

  cv::cuda::equalizeHist(resizedLeftGray_gpu, resizedEqualizedLeftGray_gpu, m_leftStream);
  cv::cuda::equalizeHist(resizedRightGray_gpu, resizedEqualizedRightGray_gpu, m_rightStream);

  m_leftRightJoinEvent.record(m_rightStream);
  m_leftStream.waitEvent(m_leftRightJoinEvent);
  //m_leftStream.waitForCompletion();

	//ConvertToGray( resizedLeft, resizedLeftGray );
	//ConvertToGray( resizedRight, resizedRightGray );
	PROFILE( "[OP] Setup" )

	{
    // workaround for no common base interface between CUDA stereo remapping algorithms that can handle the CUstream parameter to compute()
    switch (m_algorithm) {
      case 0:
        static_cast<cv::cuda::StereoBM*>(m_stereo.get())->compute(resizedLeftGray_gpu, resizedRightGray_gpu, mdisparity_gpu, m_leftStream);
        break;
      case 1:
        static_cast<cv::cuda::StereoBeliefPropagation*>(m_stereo.get())->compute(resizedLeftGray_gpu, resizedRightGray_gpu, mdisparity_gpu, m_leftStream);
        break;
      case 2:
        static_cast<cv::cuda::StereoConstantSpaceBP*>(m_stereo.get())->compute(resizedLeftGray_gpu, resizedRightGray_gpu, mdisparity_gpu, m_leftStream);
        break;
      case 3:
        static_cast<cv::cuda::StereoSGM*>(m_stereo.get())->compute(resizedLeftGray_gpu, resizedRightGray_gpu, mdisparity_gpu, m_leftStream);
        break;
    };

		//m_stereo->compute( resizedLeftGray_gpu, resizedRightGray_gpu, mdisparity_gpu );

    if (m_useDisparityFilter) {
      m_disparityFilter->apply(mdisparity_gpu, resizedLeftGray_gpu, mdisparity_filtered_gpu, m_leftStream);
      mdisparity_gpu.swap(mdisparity_filtered_gpu);
    }
/*  XXX TODO FIX
    if (m_algorithm == 0) {
      // Match type for switching between cuda::StereoBM (8UC1) and everything else (16UC1)
      mdisparity_8uc1.create(mdisparity_gpu.rows, mdisparity_gpu.cols, mdisparity_gpu.type());
      mdisparity_gpu.download(mdisparity_8uc1);
      for (size_t y = 0; y < mdisparity_8uc1.rows; ++y) {
        for (size_t x = 0; x < mdisparity_8uc1.cols; ++x) {
          mdisparity.at<uint16_t>(y,x) = mdisparity_8uc1.at<uint8_t>(y,x);
        }
      }
    }
*/

	}

	PROFILE( "[OP] Stereo Computation")

#if 0
	if ( m_useDepthBlur) {
		BlurDepths(  );
	}
	PROFILE( "[OP] Blur" )
#endif


#if 0
	{ //Process Output
		uint32_t x, y;
		for ( y = 0; y < m_iFBAlgoHeight; y++ )
		{
			// uint16_t * pxin = &m_pDisparity[y*m_iFBAlgoWidth];
			for ( x = IGNORE_EDGE_DATA_PIXELS; x < m_iFBAlgoWidth - IGNORE_EDGE_DATA_PIXELS; x++ )
			{
        uint16_t pxo = mdisparity.at<uint16_t>(y, x);
				// uint32_t pxo = pxin[x];
				int idx = y * m_iFBAlgoWidth + x;

				if ( pxo >= 0xfff0 )
				{
					pxo = 0x2020;// (x > m_iFBAlgoWidth / 2) ? 0 : 129;
					m_geoDepthMapPositions[idx * 4 + 0] = fNAN;
					m_geoDepthMapPositions[idx * 4 + 1] = fNAN;
					m_geoDepthMapPositions[idx * 4 + 2] = fNAN;
					m_geoDepthMapPositions[idx * 4 + 3] = 0;
				}
				else
				{
					//depths[x + y * m_iFBAlgoWidth] = pxin[x];
					if ( 1 )
					{
						//Update depth geometry.
						Vector4 Worldspace = TransformToWorldSpace( (float)x, (float)y, mdisparity.at<uint16_t>(y,x) );
						m_geoDepthMapPositions[idx * 4 + 0] = Worldspace[0];
						m_geoDepthMapPositions[idx * 4 + 1] = Worldspace[1];
						m_geoDepthMapPositions[idx * 4 + 2] = Worldspace[2];
						m_geoDepthMapPositions[idx * 4 + 3] = m_valids[idx];
					}
				}
			}
		}
	}
#endif

  RHICUDA::copyGpuMatToSurface(rectLeft_gpu, m_iTexture, (CUstream) m_leftStream.cudaPtr());
  RHICUDA::copyGpuMatToSurface(mdisparity_gpu, m_disparityTexture, (CUstream) m_leftStream.cudaPtr());

  //rhi()->loadTextureData(m_iTexture, kVertexElementTypeUByte4N, rectLeft.data);
  //rhi()->loadTextureData(m_disparityTexture, kVertexElementTypeShort1N, mdisparity.data);

  RHICUDA::copyGpuMatToSurface(resizedEqualizedLeftGray_gpu, m_leftGray, (CUstream) m_leftStream.cudaPtr());
  RHICUDA::copyGpuMatToSurface(resizedEqualizedRightGray_gpu, m_rightGray, (CUstream) m_leftStream.cudaPtr());
  m_leftStream.waitForCompletion();


/*
  // TODO gpu copy, or just eliminate since this is debug info
  cv::Mat resizedLeftGray, resizedRightGray;
  resizedEqualizedLeftGray_gpu.download(resizedLeftGray);
  resizedEqualizedRightGray_gpu.download(resizedRightGray);
  rhi()->loadTextureData(m_leftGray, kVertexElementTypeUByte1N, resizedLeftGray.data);
  rhi()->loadTextureData(m_rightGray, kVertexElementTypeUByte1N, resizedRightGray.data);
*/

  PROFILE( "[GL] Texture copies" )
  //rhi()->loadBufferData(m_geoDepthMapPositionBuffer, m_geoDepthMapPositions.data(), 0, m_geoDepthMapPositions.size() * sizeof(float));
  //PROFILE( "[GL] Updating output verts" )

#if DO_PROFILE
  ImGui::End();
#endif
}

void OpenCVProcess::DrawDisparityDepthMap(const FxRenderView& renderView) {
  rhi()->bindRenderPipeline(disparityDepthMapPipeline);
  rhi()->bindStreamBuffer(0, m_geoDepthMapTexcoordBuffer);

  MeshDisparityDepthMapUniformBlock ub;
  ub.modelViewProjection = renderView.viewProjectionMatrix;
  for (size_t y = 0; y < 4; ++y) {
    for (size_t x = 0; x < 4; ++x) {
      ub.R1inv[y][x] = m_R1inv[(y*4) + x];
    }
  }

  ub.Q3 = m_Q[3];
  ub.Q7 = m_Q[7];
  ub.Q11 = m_Q[11];
  ub.CameraDistanceMeters = m_CameraDistanceMeters;
  ub.mogrify = glm::vec2(MOGRIFY_X, MOGRIFY_Y);

  rhi()->loadUniformBlockImmediate(ksMeshDisparityDepthMapUniformBlock, &ub, sizeof(ub));
  rhi()->loadTexture(ksImageTex, m_iTexture);
  rhi()->loadTexture(ksDisparityTex, m_disparityTexture);

  rhi()->drawIndexedPrimitives(m_geoDepthMapTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapTristripIndexCount);
}


#if 1
void OpenCVProcess::TakeScreenshot( )
{
	struct tm timeinfo;
	time_t rawtime;
	time( &rawtime );
	localtime_r( &rawtime, &timeinfo );
	char timebuffer[128];
	std::strftime( timebuffer, sizeof( timebuffer ), "%Y%m%d %H%M%S", &timeinfo );
	std::string nowstr = timebuffer;
  cv::Mat rectLeft, rectRight;
  rectLeft_gpu.download(rectLeft);
  rectRight_gpu.download(rectRight);

  cv::Mat resizedLeftGray, resizedRightGray;
  resizedEqualizedLeftGray_gpu.download(resizedLeftGray);
  resizedEqualizedRightGray_gpu.download(resizedRightGray);


	//Make the alpha channel of the RGB maps solid.
	int sidepix = m_iFBSideWidth * m_iFBSideHeight;
	for ( int i = 0; i < sidepix; i++ )
	{
		//((uint32_t*)origLeft.data)[i] |= 0xff000000;
		//((uint32_t*)origRight.data)[i] |= 0xff000000;
		((uint32_t*)rectLeft.data)[i] |= 0xff000000;
		((uint32_t*)rectRight.data)[i] |= 0xff000000;
	}

	//stbi_write_png( (nowstr + "_Orig_RGB0.png").c_str(), m_iFBSideWidth, m_iFBSideHeight, 4, origLeft.data, m_iFBSideWidth * 4 );
	//stbi_write_png( (nowstr + "_Orig_RGB1.png").c_str(), m_iFBSideWidth, m_iFBSideHeight, 4, origRight.data, m_iFBSideWidth * 4 );
	stbi_write_png( (nowstr + "_RGB0.png").c_str(), m_iFBSideWidth, m_iFBSideHeight, 4, rectLeft.data, m_iFBSideWidth * 4 );
	stbi_write_png( (nowstr + "_RGB1.png").c_str(), m_iFBSideWidth, m_iFBSideHeight, 4, rectRight.data, m_iFBSideWidth * 4 );
	stbi_write_png( (nowstr + "_Gray0.png").c_str(), m_iFBAlgoWidth, m_iFBAlgoHeight, 1, resizedLeftGray.data, m_iFBAlgoWidth );
	stbi_write_png( (nowstr + "_Gray1.png").c_str(), m_iFBAlgoWidth, m_iFBAlgoHeight, 1, resizedRightGray.data, m_iFBAlgoWidth );

	int pxl = m_iFBAlgoWidth * m_iFBAlgoHeight;
	uint8_t * disp_px = new uint8_t[pxl];
	for ( int i = 0; i < pxl; i++ )
	{
		disp_px[i] = (uint8_t)(((uint16_t*)(mdisparity.data))[i] / 16);
	}
	stbi_write_png( (nowstr + "_Disp.png").c_str(), m_iFBAlgoWidth, m_iFBAlgoHeight, 1, disp_px, m_iFBAlgoWidth );
	delete[] disp_px;
}
#endif
