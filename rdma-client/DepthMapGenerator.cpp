#include "imgui.h"
#include "DepthMapGenerator.h"
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
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

#include "stb/stb_image_write.h"

#define NUM_DISP 128 //Max disparity. Must be in {64, 128, 256} for CUDA algorithms.
#define MOGRIFY_X 4
#define MOGRIFY_Y 4



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

static double OGGetAbsoluteTime() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return ((double)tv.tv_usec)/1000000. + (tv.tv_sec);
}

glm::mat4 Matrix4FromCVMatrix(cv::Mat matin) {
  glm::mat4 out(1.0f);
  for (int y = 0; y < matin.rows; y++) {
    for (int x = 0; x < matin.cols; x++) {
      out[y][x] = (float)matin.at<double>(y, x);
    }
  }
  return out;
}

DepthMapGenerator::DepthMapGenerator(CameraSystem* cs, size_t viewIdx) :
    m_cameraSystem(cs), m_viewIdx(viewIdx), m_iProcFrames(0), m_iFramesSinceFPS(0), m_dTimeOfLastFPS(0), m_leftRightJoinEvent(cv::cuda::Event::DISABLE_TIMING), m_enableProfiling(false) {

  if (!disparityDepthMapPipeline) {
    disparityDepthMapPipeline = rhi()->compileRenderPipeline("shaders/meshDisparityDepthMap.vtx.glsl", "shaders/meshTexture.frag.glsl", RHIVertexLayout({
        RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "textureCoordinates", 0, sizeof(float) * 4)
      }), kPrimitiveTopologyTriangleStrip);
  }


  // Initial algorithm settings

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



  CameraSystem::View& v = m_cameraSystem->viewAtIndex(m_viewIdx);
  CameraSystem::Camera& cL = m_cameraSystem->cameraAtIndex(v.cameraIndices[0]);
  CameraSystem::Camera& cR = m_cameraSystem->cameraAtIndex(v.cameraIndices[1]);

  // TODO validate CameraSystem:::updateViewStereoDistortionParameters against the distortion map initialization code here
  cv::Size imageSize = cv::Size(m_cameraSystem->cameraProvider()->streamWidth(), m_cameraSystem->cameraProvider()->streamHeight());

  {
    cv::Mat m1, m2;
    cv::initUndistortRectifyMap(cL.intrinsicMatrix, cL.distCoeffs, v.stereoRectification[0], v.stereoProjection[0], imageSize, CV_32F, m1, m2);
    m_leftMap1_gpu.upload(m1); m_leftMap2_gpu.upload(m2);
    cv::initUndistortRectifyMap(cR.intrinsicMatrix, cR.distCoeffs, v.stereoRectification[1], v.stereoProjection[1], imageSize, CV_32F, m1, m2);
    m_rightMap1_gpu.upload(m1); m_rightMap2_gpu.upload(m2);

  }
  m_R1 = Matrix4FromCVMatrix(v.stereoRectification[0]);
  m_R1inv = glm::inverse(m_R1);
  m_Q = Matrix4FromCVMatrix(v.stereoDisparityToDepth);

  m_CameraDistanceMeters = glm::length(glm::vec3(v.stereoTranslation.at<double>(0), v.stereoTranslation.at<double>(1), v.stereoTranslation.at<double>(2)));


  m_iFBSideWidth = m_cameraSystem->cameraProvider()->streamWidth();
  m_iFBSideHeight = m_cameraSystem->cameraProvider()->streamHeight();

  m_iFBAlgoWidth = m_iFBSideWidth / MOGRIFY_X;
  m_iFBAlgoHeight = m_iFBSideHeight / MOGRIFY_Y;

  m_iTexture = rhi()->newTexture2D(m_iFBSideWidth, m_iFBSideHeight, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
  m_disparityTexture = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor(kSurfaceFormat_R16));
  m_leftGray = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));
  m_rightGray = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));

  //Set up what matrices we can to prevent dynamic memory allocation.

  origLeft_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);
  origRight_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);

  rectLeft_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);
  rectRight_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);

  resizedLeft_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth, CV_8UC4);
  resizedRight_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth, CV_8UC4);

  resizedLeftGray_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);
  resizedRightGray_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);

  resizedEqualizedLeftGray_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);
  resizedEqualizedRightGray_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth + NUM_DISP, CV_8U);

  mdisparity = cv::Mat(m_iFBAlgoHeight, m_iFBAlgoWidth, CV_16S);
  mdisparity_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth, CV_16S);


  // Set up geometry depth map data

  // Set up geometry map texcoord and index buffers
  // (borrowed from camera_app.cpp)
  {
    { // Texcoord and position buffers
      std::vector<float> depth_tc;
      int uiDepthVertCount = m_iFBAlgoWidth * m_iFBAlgoHeight;
      depth_tc.resize(uiDepthVertCount * 4);
      for (int y = 0; y < m_iFBAlgoHeight; y++) {
        for (int x = 0; x < m_iFBAlgoWidth; x++) {
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
      depth_ia.resize(m_iFBAlgoWidth * dmym1 * 2);
      //int uiDepthIndexCount = (unsigned int)depth_ia.size();
      for (int y = 0; y < dmym1; y++) {
        for (int x = 0; x < m_iFBAlgoWidth; x++) {
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
      depth_ia_lines.resize(m_iFBAlgoWidth * dmym1 * 2);
      //int uiDepthIndexCountLines = (unsigned int)depth_ia_lines.size();

      for (int y = 0; y < dmym1; y++) {
        for (int x = 0; x < m_iFBAlgoWidth; x += 2) {
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
}

DepthMapGenerator::~DepthMapGenerator() {
}

void ConvertToGray(cv::InputArray src, cv::OutputArray dst) {
  //You can do this the OpenCV way, but my alternative transform seems more successful.
  cv::Mat msrc(src.getMat());
  cv::Mat mdst(dst.getMat());

  int w = msrc.cols;
  int h = msrc.rows;

  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      uint32_t inx = msrc.at<uint32_t>(y, x);

      int r = (inx >> 0) & 0xff;
      int g = (inx >> 8) & 0xff;
      int b = (inx >> 16) & 0xff;
      mdst.at<uint8_t>(y, x) = (uint8_t)((r + g + b) / 3);

    }
  }
}

/*
Vector4 DepthMapGenerator::TransformToLocalSpace(float x, float y, int disp)
{
  float fDisp = (float) disp / 16.f; //  16-bit fixed-point disparity map (where each disparity value has 4 fractional bits)

  float lz = m_Q[11] * m_CameraDistanceMeters / (fDisp * MOGRIFY_X);
  float ly = -(y * MOGRIFY_Y + m_Q[7]) / m_Q[11];
  float lx = (x * MOGRIFY_X + m_Q[3]) / m_Q[11];
  lx *= lz;
  ly *= lz;
  lz *= -1;
  return m_R1inv * Vector4(lx, ly, lz, 1.0);
}
*/

void DepthMapGenerator::processFrame() {
  m_setupTimeMs = 0;
  m_algoTimeMs = 0;
  m_filterTimeMs = 0;
  m_copyTimeMs = 0;

  if (m_enableProfiling) {
    try {
      m_setupTimeMs = cv::cuda::Event::elapsedTime(m_setupStartEvent, m_algoStartEvent);
      m_algoTimeMs = cv::cuda::Event::elapsedTime(m_algoStartEvent, m_filterStartEvent);
      m_filterTimeMs = cv::cuda::Event::elapsedTime(m_filterStartEvent, m_copyStartEvent);
      m_copyTimeMs = cv::cuda::Event::elapsedTime(m_copyStartEvent, m_processingFinishedEvent);
    } catch (...) {
      // cv::cuda::Event::elapsedTime will throw if the event is not ready; just skip reading the event timers in that case.
    }
  }

  m_iProcFrames++;
  m_iFramesSinceFPS++;

  double Start = OGGetAbsoluteTime();
  if (Start >= m_dTimeOfLastFPS + 1) {
    if (Start - m_dTimeOfLastFPS < 4)
      m_dTimeOfLastFPS++;
    else
      m_dTimeOfLastFPS = Start;
    m_iFPS = m_iFramesSinceFPS;
    m_iFramesSinceFPS = 0;
  }


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

  // Setup: Copy input camera surfaces to CV GpuMats, resize/grayscale/normalize

  RHICUDA::copySurfaceToGpuMat(m_cameraSystem->cameraProvider()->rgbTexture(m_cameraSystem->viewAtIndex(m_viewIdx).cameraIndices[0]), origLeft_gpu/*, m_leftStream*/);
  RHICUDA::copySurfaceToGpuMat(m_cameraSystem->cameraProvider()->rgbTexture(m_cameraSystem->viewAtIndex(m_viewIdx).cameraIndices[1]), origRight_gpu/*, m_rightStream*/);

  if (m_enableProfiling) {
    m_setupStartEvent.record(m_leftStream);
  }

  cv::cuda::remap(origLeft_gpu, rectLeft_gpu, m_leftMap1_gpu, m_leftMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue=*/ cv::Scalar(), m_leftStream);
  cv::cuda::remap(origRight_gpu, rectRight_gpu, m_rightMap1_gpu, m_rightMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue=*/ cv::Scalar(), m_rightStream);

  cv::cuda::resize(rectLeft_gpu, resizedLeft_gpu, cv::Size(m_iFBAlgoWidth, m_iFBAlgoHeight), 0, 0, cv::INTER_LINEAR, m_leftStream);
  cv::cuda::resize(rectRight_gpu, resizedRight_gpu, cv::Size(m_iFBAlgoWidth, m_iFBAlgoHeight), 0, 0, cv::INTER_LINEAR, m_rightStream);

  cv::cuda::cvtColor(resizedLeft_gpu, resizedLeftGray_gpu, cv::COLOR_BGRA2GRAY, 0, m_leftStream);
  cv::cuda::cvtColor(resizedRight_gpu, resizedRightGray_gpu, cv::COLOR_BGRA2GRAY, 0, m_rightStream);

  cv::cuda::equalizeHist(resizedLeftGray_gpu, resizedEqualizedLeftGray_gpu, m_leftStream);
  cv::cuda::equalizeHist(resizedRightGray_gpu, resizedEqualizedRightGray_gpu, m_rightStream);

  m_leftRightJoinEvent.record(m_rightStream);
  m_leftStream.waitEvent(m_leftRightJoinEvent);

  if (m_enableProfiling) {
    m_algoStartEvent.record(m_leftStream);
  }

  // Disparity computation

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

    //m_stereo->compute(resizedLeftGray_gpu, resizedRightGray_gpu, mdisparity_gpu);

    if (m_enableProfiling) {
      m_filterStartEvent.record(m_leftStream);
    }
   
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

  if (m_enableProfiling) {
    m_copyStartEvent.record(m_leftStream);
  }

  // Copy results to render surfaces

  RHICUDA::copyGpuMatToSurface(rectLeft_gpu, m_iTexture, m_leftStream);
  RHICUDA::copyGpuMatToSurface(mdisparity_gpu, m_disparityTexture, m_leftStream);

  RHICUDA::copyGpuMatToSurface(resizedEqualizedLeftGray_gpu, m_leftGray, m_leftStream);
  RHICUDA::copyGpuMatToSurface(resizedEqualizedRightGray_gpu, m_rightGray, m_leftStream);
  if (m_enableProfiling) {
    m_processingFinishedEvent.record(m_leftStream);
  }

  m_leftStream.waitForCompletion();
}

void DepthMapGenerator::renderDisparityDepthMap(const FxRenderView& renderView) {
  rhi()->bindRenderPipeline(disparityDepthMapPipeline);
  rhi()->bindStreamBuffer(0, m_geoDepthMapTexcoordBuffer);

  MeshDisparityDepthMapUniformBlock ub;
  ub.modelViewProjection = renderView.viewProjectionMatrix;
  ub.R1inv = m_R1inv;

  ub.Q3 = m_Q[0][3];
  ub.Q7 = m_Q[1][3];
  ub.Q11 = m_Q[2][3];
  ub.CameraDistanceMeters = m_CameraDistanceMeters;
  ub.mogrify = glm::vec2(MOGRIFY_X, MOGRIFY_Y);

  rhi()->loadUniformBlockImmediate(ksMeshDisparityDepthMapUniformBlock, &ub, sizeof(ub));
  rhi()->loadTexture(ksImageTex, m_iTexture);
  rhi()->loadTexture(ksDisparityTex, m_disparityTexture);

  rhi()->drawIndexedPrimitives(m_geoDepthMapTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapTristripIndexCount);
}

void DepthMapGenerator::renderIMGUI() {
  ImGui::PushID(this);

  ImGui::Text("Stereo Algorithm");
  m_didChangeSettings |= ImGui::RadioButton("BM", &m_algorithm, 0);
  m_didChangeSettings |= ImGui::RadioButton("BeliefPropagation", &m_algorithm, 1);
  m_didChangeSettings |= ImGui::RadioButton("ConstantSpaceBeliefPropagation", &m_algorithm, 2);
  m_didChangeSettings |= ImGui::RadioButton("SGM", &m_algorithm, 3);

  switch (m_algorithm) {
    case 0: // StereoBM
      m_didChangeSettings |= ImGui::InputInt("Block Size (odd)", &m_sbmBlockSize, /*step=*/2);
      break;
    case 2: // StereoConstantSpaceBP
      m_didChangeSettings |= ImGui::SliderInt("nr_plane", &m_scsbpNrPlane, 1, 16);
      // fallthrough for shared parameters
    case 1: // StereoBeliefPropagation
      m_didChangeSettings |= ImGui::SliderInt("SBP Iterations", &m_sbpIterations, 1, 8);
      m_didChangeSettings |= ImGui::SliderInt("SBP Levels", &m_sbpLevels, 1, 8);
      break;
    case 3: // StereoSGM
      m_didChangeSettings |= ImGui::SliderInt("SGM P1", &m_sgmP1, 1, 255);
      m_didChangeSettings |= ImGui::SliderInt("SGM P2", &m_sgmP2, 1, 255);
      m_didChangeSettings |= ImGui::SliderInt("SGM Uniqueness Ratio", &m_sgmUniquenessRatio, 5, 15);
      break;
  };

  m_didChangeSettings |= ImGui::Checkbox("Disparity filter (GPU)", &m_useDisparityFilter);

  if (m_useDisparityFilter) {
    m_didChangeSettings |= ImGui::SliderInt("Filter Radius (odd)", &m_disparityFilterRadius, 1, 9);
    m_didChangeSettings |= ImGui::SliderInt("Filter Iterations", &m_disparityFilterIterations, 1, 8);
  }

  ImGui::Checkbox("CUDA Profiling", &m_enableProfiling);
  if ((m_iProcFrames > 1) && m_enableProfiling) {
    // TODO use cuEventElapsedTime and skip if the return is not CUDA_SUCCESS --
    // cv::cuda::Event::elapsedTime throws an exception on CUDA_ERROR_NOT_READY
    //float f;
    //if (cuEventElapsedTime(&f, m_setupStartEvent.event
    ImGui::Text("Setup: %.3fms", m_setupTimeMs);
    ImGui::Text("Algo: %.3fms", m_algoTimeMs);
    ImGui::Text("Filter: %.3fms", m_filterTimeMs);
    ImGui::Text("Copy: %.3fms", m_copyTimeMs);
  }

  ImGui::Text("Frames: %5d; %3d FPS", m_iProcFrames, m_iFPS);

  ImGui::PopID();
}

