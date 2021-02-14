#include "OpenCVProcess.h"
#include "common/CameraSystem.h"
#include "RDMACameraProvider.h"
#include "rhi/RHI.h"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <time.h>
#include <sys/time.h>
#include <thread>
#include "Matrices.h"
#include "Vectors.h"

#include "stb/stb_image_write.h"

#define NUM_DISP 128 //Max disparity.

inline double * CoPTr( const std::initializer_list<double>& d ) { return (double*)d.begin(); }
#define DO_FISHEYE 1

#define DO_PROFILE 1
#if DO_PROFILE
#define PROFILE( x ) { double Now = OGGetAbsoluteTime(); printf("\x1b[2K%27s: %.3fms\n", x, (Now-Start)*1000.0); Start = Now; }
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
	, m_iHasFrameForUpdate( 0 )
	, m_iDoneFrameOutput( 0 )
	, m_pthread( 0 )
	, m_bQuitThread( false )
	, m_bScreenshotNext( 0 )
  , m_useDepthBlur(true)
{
	fNAN = nanf( "" );


/*
		else if ( m_iCurrentStereoAlgorithm == 2 )
		{
			m_stereo = cv::StereoSGBM::create( 0, NUM_DISP, 15,
				0, 0, 0,
				4, 5,
				200, 1,
				cv::StereoSGBM::MODE_SGBM );
		}
*/

  m_didChangeSettings = false;
  m_algorithm = 1;
  m_blockSize = 15;
  m_preFilterCap = 4;
  m_uniquenessRatio = 5;
  m_speckleWindowSize = 200;
  m_speckleRange = 1;


}

OpenCVProcess::~OpenCVProcess()
{
	m_bQuitThread = true;
	if ( m_pthread )
	{
		m_pthread->join();
	}
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

  rectLeft = cv::Mat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);

  rectLeft_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);
  rectRight_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);

  resizedLeft_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth, CV_8UC4);
  resizedRight_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth, CV_8UC4);

  resizedLeftGray_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);
  resizedRightGray_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);

  resizedEqualizedLeftGray_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);
  resizedEqualizedRightGray_gpu = cv::cuda::GpuMat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);

  resizedLeftGray = cv::Mat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);
  resizedRightGray = cv::Mat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);

	mdisparity = cv::Mat( m_iFBAlgoHeight, m_iFBAlgoWidth, CV_16S );
	mdisparity_expanded = cv::Mat( m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_16S );


  // Set up geometry depth map data
  m_geoDepthMapPositions.resize(m_iFBAlgoWidth  * m_iFBAlgoHeight * 4);
  m_geoDepthMapPositionBuffer = rhi()->newEmptyBuffer(m_geoDepthMapPositions.size() * sizeof(float), kBufferUsageCPUWriteOnly);

  // Set up geometry map texcoord and index buffers
  // (borrowed from camera_app.cpp)
	{
    { // Texcoord and position buffers
      std::vector<float>& depth_vc = m_geoDepthMapPositions;
      std::vector<float> depth_tc;
      int uiDepthVertCount = m_iFBAlgoWidth * m_iFBAlgoHeight;
      depth_tc.resize( uiDepthVertCount * 2 );
      depth_vc.resize( uiDepthVertCount * 4 );
      for ( int y = 0; y < m_iFBAlgoHeight; y++ ) {
        for ( int x = 0; x < m_iFBAlgoWidth; x++ ) {
          depth_tc[(x + y * m_iFBAlgoWidth) * 2 + 0] = 1.0f * x / (m_iFBAlgoWidth - 1);
          depth_tc[(x + y * m_iFBAlgoWidth) * 2 + 1] = 1.0f * y / (m_iFBAlgoHeight - 1);

          float cx = 10.0f * x / (m_iFBAlgoWidth - 1) - 5;
          float cy = 10.0f * y / (m_iFBAlgoHeight - 1) - 5;
          float sincR = sqrt( cx * cx + cy * cy );
          float sinc = sin( sincR ) / sincR;
          depth_vc[(x + y * m_iFBAlgoWidth) * 4 + 0] = 2.0f * x / (m_iFBAlgoHeight - 1) - 1.0f;
          depth_vc[(x + y * m_iFBAlgoWidth) * 4 + 1] = 1 ? nanf( "" ) : sinc;
          depth_vc[(x + y * m_iFBAlgoWidth) * 4 + 2] = 2.0f * y / (m_iFBAlgoHeight - 1) - 1.0f;
          depth_vc[(x + y * m_iFBAlgoWidth) * 4 + 3] = 1.0;
        }
      }
      m_geoDepthMapTexcoordBuffer = rhi()->newBufferWithContents(depth_tc.data(), depth_tc.size() * sizeof(float), kBufferUsageCPUWriteOnly);
      m_geoDepthMapPositionBuffer = rhi()->newBufferWithContents(depth_vc.data(), depth_vc.size() * sizeof(float), kBufferUsageCPUWriteOnly);
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

  // Start background processing thread
	m_pthread = new std::thread( &OpenCVProcess::Thread, this );

  return true;
}

void OpenCVProcess::Thread()
{
	while ( !m_bQuitThread )
	{
		if ( m_iHasFrameForUpdate == 2 )
		{
      OpenCVAppUpdate();
			m_iHasFrameForUpdate = 0;
		}
		OGUSleep( 1000 );
	}
}

void OpenCVProcess::Prerender()
{
	if ( m_iDoneFrameOutput )
	{
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

    rhi()->loadTextureData(m_iTexture, kVertexElementTypeUByte4N, rectLeft.data);
    rhi()->loadTextureData(m_disparityTexture, kVertexElementTypeUShort1N, mdisparity.data);
    rhi()->loadTextureData(m_leftGray, kVertexElementTypeUByte1N, resizedLeftGray.data);
    rhi()->loadTextureData(m_rightGray, kVertexElementTypeUByte1N, resizedRightGray.data);

		PROFILE( "[GL] Updating output texture" )
    rhi()->loadBufferData(m_geoDepthMapPositionBuffer, m_geoDepthMapPositions.data(), 0, m_geoDepthMapPositions.size() * sizeof(float));
		PROFILE( "[GL] Updating output verts" )
		m_iDoneFrameOutput = 0;
	}

	if ( m_iHasFrameForUpdate == 1 )
	{
#if 0 // OLD readback
		double Start = OGGetAbsoluteTime();
		glBindBuffer( GL_PIXEL_PACK_BUFFER, m_iPBOids[0] );
		m_pFrameBuffer = (GLubyte*)glMapBuffer( GL_PIXEL_PACK_BUFFER, GL_READ_ONLY );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
		PROFILE( "[GL] Readback" )
#endif
		m_iHasFrameForUpdate = 2;
	}

	if ( m_iHasFrameForUpdate == 0 )
	{
		//double Start = OGGetAbsoluteTime();

#if DO_PROFILE
		printf("\x1b[1;1f" );
		printf("\x1b[2K\x1b[34mFrames: %5d; %3d FPS\x1b[0m\n", m_iProcFrames, m_iFPS );
		printf("\x1b[32mGreen FG Test\x1b[0m\n" );
		printf("\x1b[31mRed FG Test\x1b[0m\n" );
		printf("\x1b[0m" );
#endif

#if 0 // OLD readback
		//This uses OpenGL to read back the pixels.  It seems to be MUCH faster than the DX alternative inside SteamVR.
		vr::EVRTrackedCameraError ce = vr::VRTrackedCamera()->GetVideoStreamTextureGL( m_pCamera, DO_FISHEYE ? vr::VRTrackedCameraFrameType_Distorted : vr::VRTrackedCameraFrameType_Undistorted, &m_iGLimback, &m_lastFrameHeader, sizeof( m_lastFrameHeader ) );
		m_lastFrameHeaderMatrix = ConvertSteamVRMatrixToMatrix4( m_lastFrameHeader.trackedDevicePose.mDeviceToAbsoluteTracking );

		PROFILE( "[GL] GetVideoStreamTexture" )
		glFinish();
		PROFILE( "[GL] Flush" )

		glBindFramebuffer( GL_FRAMEBUFFER, m_iGLfrback );
		glFramebufferTexture( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_iGLimback, 0 );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, m_iPBOids[0] );
		if ( m_pFrameBuffer ) glUnmapBuffer( GL_PIXEL_PACK_BUFFER );
		glReadPixels( 0, 0, m_lastFrameHeader.nWidth, m_lastFrameHeader.nHeight, GL_RGBA, GL_UNSIGNED_BYTE, 0 );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
		glBindFramebuffer( GL_FRAMEBUFFER, 0 );
		PROFILE( "[GL] PBO read is setup." )
#else
		//m_lastFrameHeaderMatrix = glm::mat4(1.0f); // identity matrix; TODO actual transform
		m_lastFrameHeaderMatrix.identity(); // identity matrix; TODO actual transform
#endif

    m_iHasFrameForUpdate = 1;
	}


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

	Matrix4 mlefteye = m_lastFrameHeaderMatrix;
	Vector4 placerelview( local );
	Vector4 Worldview = mlefteye * (placerelview);
	return Worldview;
}

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
      uint16_t pxi = mdisparity.at<uint16_t>(y,x);
			int idx = (y * m_iFBAlgoWidth) + x;
			if ( pxi == 0 || pxi >= m_iFBAlgoWidth * 16 )
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

void OpenCVProcess::OpenCVAppUpdate()
{
	double Start = OGGetAbsoluteTime();

  if ((!m_stereo) || m_didChangeSettings) {
    m_blockSize |= 1; // enforce odd blockSize

    switch (m_algorithm) {
      default:
        m_algorithm = 0;
      case 0:
        m_stereo = cv::StereoSGBM::create(0, NUM_DISP, m_blockSize,
          0, 0, 0,
          m_preFilterCap, m_uniquenessRatio,
          m_speckleWindowSize, m_speckleRange,
          cv::StereoSGBM::MODE_SGBM );
        break;
      case 1: {
        m_stereo = cv::StereoBM::create(NUM_DISP, m_blockSize);
      } break;

    };
    


  }

	origLeft_gpu.upload(m_cameraProvider->cvMat(m_cameraSystem->viewAtIndex(m_viewIdx).cameraIndices[0]));
	origRight_gpu.upload(m_cameraProvider->cvMat(m_cameraSystem->viewAtIndex(m_viewIdx).cameraIndices[1]));

	cv::cuda::remap( origLeft_gpu, rectLeft_gpu, m_leftMap1_gpu, m_leftMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT );
	cv::cuda::remap( origRight_gpu, rectRight_gpu, m_rightMap1_gpu, m_rightMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT );
  rectLeft_gpu.download(rectLeft);

	cv::cuda::resize( rectLeft_gpu, resizedLeft_gpu, cv::Size( m_iFBAlgoWidth, m_iFBAlgoHeight ) );
	cv::cuda::resize( rectRight_gpu, resizedRight_gpu, cv::Size( m_iFBAlgoWidth, m_iFBAlgoHeight ) );

  cv::cuda::cvtColor( resizedLeft_gpu, resizedLeftGray_gpu, cv::COLOR_BGRA2GRAY );
  cv::cuda::cvtColor( resizedRight_gpu, resizedRightGray_gpu, cv::COLOR_BGRA2GRAY );

  cv::cuda::equalizeHist(resizedLeftGray_gpu, resizedEqualizedLeftGray_gpu);
  cv::cuda::equalizeHist(resizedRightGray_gpu, resizedEqualizedRightGray_gpu);

  resizedEqualizedLeftGray_gpu.download(resizedLeftGray);
  resizedEqualizedRightGray_gpu.download(resizedRightGray);

	//ConvertToGray( resizedLeft, resizedLeftGray );
	//ConvertToGray( resizedRight, resizedRightGray );
	PROFILE( "[OP] Setup" )

	{
		m_stereo->compute( resizedLeftGray, resizedRightGray, mdisparity_expanded );

		for (uint32_t y = 0; y < m_iFBAlgoHeight; y++) {
			uchar* indata = mdisparity_expanded.ptr(y);
			uchar* outdata = mdisparity.ptr(y);
      memcpy(outdata, indata, sizeof(uint16_t) * m_iFBAlgoWidth);
		}
	}

	if ( m_bScreenshotNext )
	{
		TakeScreenshot();
		m_bScreenshotNext = false;
	}

	PROFILE( "[OP] Stereo Computation")

	if ( m_useDepthBlur) {
		BlurDepths(  );
	}
	PROFILE( "[OP] Blur" )


#if 1
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
#else

  {
    cv::Mat outImage(m_iFBAlgoHeight, m_iFBAlgoWidth, CV_32FC3);
    cv::reprojectImageTo3D(mdisparity, outImage, m_cameraSystem->viewAtIndex(m_viewIdx).stereoDisparityToDepth, /*handleMissingValues=*/true);
    for (size_t y = 0; y < m_iFBAlgoHeight; ++y) {
      for (size_t x = 0; x < m_iFBAlgoWidth; ++x) {
				size_t idx = (y * m_iFBAlgoWidth) + x;
        float* p = (float*) (outImage.ptr(y, x));
        m_geoDepthMapPositions[(idx * 4) + 0] = p[0];
        m_geoDepthMapPositions[(idx * 4) + 1] = p[1];
        m_geoDepthMapPositions[(idx * 4) + 2] = p[2];
        m_geoDepthMapPositions[(idx * 4) + 3] = 1.0f;
      }
    }

  }

#endif

	m_iDoneFrameOutput = 1;

	PROFILE( "[OP] Process" )


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
  cv::Mat rectRight;
  rectRight_gpu.download(rectRight);

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
