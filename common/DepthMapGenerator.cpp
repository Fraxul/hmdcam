#include "imgui.h"
#include "common/DepthMapGenerator.h"
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/glmCvInterop.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/RHICVInterop.h"
#include "rhi/gl/GLCommon.h"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <thread>

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
  float disparityPrescale;
  int disparityTexLevels;

  glm::vec2 trim_minXY;
  glm::vec2 trim_maxXY;
};

RHIRenderPipeline::ptr disparityMipPipeline;
FxAtomicString ksDisparityMipUniformBlock("DisparityMipUniformBlock");
struct DisparityMipUniformBlock {
  uint32_t sourceLevel;
  float pad2, pad3, pad4;
};

static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

static inline float deltaTimeMs(uint64_t startTimeNs, uint64_t endTimeNs) {
  return static_cast<float>(endTimeNs - startTimeNs) / 1000000.0f;
}

DepthMapGenerator::DepthMapGenerator(CameraSystem* cs, SHMSegment<DepthMapSHM>* shm) :
    m_cameraSystem(cs), m_depthMapSHM(shm), m_enableProfiling(false), m_populateDebugTextures(false) {


  if (!disparityDepthMapPipeline) {
    RHIRenderPipelineDescriptor rpd;
    rpd.primitiveTopology = kPrimitiveTopologyTriangleStrip;
    rpd.primitiveRestartEnabled = true;

    disparityDepthMapPipeline = rhi()->compileRenderPipeline(
      rhi()->compileShader(RHIShaderDescriptor("shaders/meshDisparityDepthMap.vtx.glsl", "shaders/meshTexture.frag.glsl", RHIVertexLayout({
        RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "textureCoordinates", 0, sizeof(float) * 4)
      }))), rpd);
  }

  if (!disparityMipPipeline) {
    disparityMipPipeline = rhi()->compileRenderPipeline("shaders/ndcQuad.vtx.glsl", "shaders/disparityMip.frag.glsl", ndcQuadVertexLayout, kPrimitiveTopologyTriangleStrip);
  }


  m_didChangeSettings = true; // force initial algorithm setup

  // Initial algorithm settings
  m_algorithm = 2;
  m_useDisparityFilter = true;
  m_disparityFilterRadius = 3;
  m_disparityFilterIterations = 1;

  // cuda::StereoBM, algorithm 0
  m_sbmBlockSize = 19; // must be odd

  // cuda::StereoConstantSpaceBP, algorithm 1
  m_sbpIterations = 5;
  m_sbpLevels = 5;
  m_scsbpNrPlane = 4;

  // cuda::StereoSGM, algorithm 2
  m_sgmP1 = 10;
  m_sgmP2 = 120;
  m_sgmUniquenessRatio = 5; // 5-15
  m_sgmUseHH4 = true;

  // Render settings
  m_trimLeft = 8;
  m_trimTop = 8;
  m_trimRight = 8;
  m_trimBottom = 8;


  // Common data
  m_iFBSideWidth = m_cameraSystem->cameraProvider()->streamWidth();
  m_iFBSideHeight = m_cameraSystem->cameraProvider()->streamHeight();

  m_iFBAlgoWidth = m_iFBSideWidth / MOGRIFY_X;
  m_iFBAlgoHeight = m_iFBSideHeight / MOGRIFY_Y;

  // Create depth map geometry buffers
  {
    { // Texcoord and position buffers
      std::vector<float> depth_tc;
      int uiDepthVertCount = m_iFBAlgoWidth * m_iFBAlgoHeight;
      depth_tc.resize(uiDepthVertCount * 4);
      for (int y = 0; y < m_iFBAlgoHeight; y++) {
        for (int x = 0; x < m_iFBAlgoWidth; x++) {
          // xy is image texture coordinates (0...1)
          depth_tc[(x + y * m_iFBAlgoWidth) * 4 + 0] = static_cast<float>(x) / static_cast<float>(m_iFBAlgoWidth - 1);
          depth_tc[(x + y * m_iFBAlgoWidth) * 4 + 1] = static_cast<float>(y) / static_cast<float>(m_iFBAlgoHeight - 1);
          // zw is disparity map coordinates (texels)
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
      depth_ia.reserve((m_iFBAlgoWidth * dmym1 * 2) + dmym1);
      //int uiDepthIndexCount = (unsigned int)depth_ia.size();
      for (int y = 0; y < dmym1; y++) {
        if (y != 0)
          depth_ia.push_back(0xffffffff); // strip-restart

        for (int x = 0; x < m_iFBAlgoWidth; x++) {
          depth_ia.push_back(x + ( y      * (m_iFBAlgoWidth)));
          depth_ia.push_back(x + ((y + 1) * (m_iFBAlgoWidth)));
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

  // Per-view data
  m_viewData.resize(m_cameraSystem->views());

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    CameraSystem::View& v = m_cameraSystem->viewAtIndex(viewIdx);
    ViewData& vd = m_viewData[viewIdx];

    vd.m_isStereoView = v.isStereo;
    if (!vd.m_isStereoView)
      continue; // Not applicable for mono views

    vd.m_isVerticalStereo = v.isVerticalStereo();

    CameraSystem::Camera& cL = m_cameraSystem->cameraAtIndex(v.cameraIndices[0]);
    CameraSystem::Camera& cR = m_cameraSystem->cameraAtIndex(v.cameraIndices[1]);

    // TODO validate CameraSystem:::updateViewStereoDistortionParameters against the distortion map initialization code here
    cv::Size imageSize = cv::Size(m_cameraSystem->cameraProvider()->streamWidth(), m_cameraSystem->cameraProvider()->streamHeight());

    {
      cv::Mat m1, m2;
      cv::initUndistortRectifyMap(cL.intrinsicMatrix, cL.distCoeffs, v.stereoRectification[0], v.stereoProjection[0], imageSize, CV_32F, m1, m2);
      vd.m_leftMap1_gpu.upload(m1); vd.m_leftMap2_gpu.upload(m2);
      cv::initUndistortRectifyMap(cR.intrinsicMatrix, cR.distCoeffs, v.stereoRectification[1], v.stereoProjection[1], imageSize, CV_32F, m1, m2);
      vd.m_rightMap1_gpu.upload(m1); vd.m_rightMap2_gpu.upload(m2);

    }
    vd.m_R1inv = glm::inverse(glmMat4FromCVMatrix(v.stereoRectification[0]));
    vd.m_Q = glmMat4FromCVMatrix(v.stereoDisparityToDepth);

    vd.m_CameraDistanceMeters = glm::length(glm::vec3(v.stereoTranslation.at<double>(0), v.stereoTranslation.at<double>(1), v.stereoTranslation.at<double>(2)));


    vd.m_iTexture = rhi()->newTexture2D(m_iFBSideWidth, m_iFBSideHeight, RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    // vd.m_disparityTexture = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor(kSurfaceFormat_R16i));
    vd.m_leftGray = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));
    vd.m_rightGray = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor(kSurfaceFormat_R8));

    //Set up what matrices we can to prevent dynamic memory allocation.

    vd.origLeft_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);
    vd.origRight_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);

    vd.rectLeft_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);
    vd.rectRight_gpu = cv::cuda::GpuMat(m_iFBSideHeight, m_iFBSideWidth, CV_8UC4);

    vd.resizedLeft_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth, CV_8UC4);
    vd.resizedRight_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth, CV_8UC4);

    vd.resizedLeftGray_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth, CV_8U);
    vd.resizedRightGray_gpu = cv::cuda::GpuMat(m_iFBAlgoHeight, m_iFBAlgoWidth, CV_8U);

    if (vd.m_isVerticalStereo) {
      vd.resizedTransposedLeftGray_gpu = cv::cuda::GpuMat(m_iFBAlgoWidth, m_iFBAlgoHeight, CV_8U);
      vd.resizedTransposedRightGray_gpu = cv::cuda::GpuMat(m_iFBAlgoWidth, m_iFBAlgoHeight, CV_8U);
    }
  }

}

DepthMapGenerator::~DepthMapGenerator() {
  delete m_depthMapSHM;
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
  m_copyTimeMs = 0;

  if (m_enableProfiling) {
    try {
      m_setupTimeMs = cv::cuda::Event::elapsedTime(m_setupStartEvent, m_setupFinishedEvent);
      m_copyTimeMs = cv::cuda::Event::elapsedTime(m_copyStartEvent, m_processingFinishedEvent);
    } catch (...) {
      // cv::cuda::Event::elapsedTime will throw if the event is not ready; just skip reading the event timers in that case.
    }
  }

  if (m_didChangeSettings) {
    m_sbmBlockSize |= 1; // enforce odd blockSize
    m_disparityFilterRadius |= 1; // enforce odd filter size

    switch (m_algorithm) {
      default:
        m_algorithm = 0;
      case 0:
        // uses CV_8UC1 disparity
        m_disparityPrescale = 1.0f; // / 16.0f;
        break;
      case 1: {
        if (m_sbpLevels > 4) // more than 4 levels seems to trigger an internal crash
          m_sbpLevels = 4;
        m_disparityPrescale = 1.0f; // / 16.0f;
      } break;
      case 2:
        m_disparityPrescale = 1.0f / 16.0f; // TODO: not sure if this is correct -- matches results from CSBP, roughly.
        break;
    };

    for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
      ViewData& vd = m_viewData[viewIdx];
      if (!vd.m_isStereoView)
        continue;

      vd.m_disparityTextureMipTargets.clear();
      if (m_algorithm == 0) {
        // CV_8uc1 disparity map type for StereoBM
        vd.m_disparityTexture = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor::mipDescriptor(kSurfaceFormat_R8i));
      } else {
        // CV_16sc1 disparity map type for everything else
        vd.m_disparityTexture = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor::mipDescriptor(kSurfaceFormat_R16i));
      }

      for (uint32_t level = 0; level < vd.m_disparityTexture->mipLevels(); ++level) {
        vd.m_disparityTextureMipTargets.push_back(rhi()->compileRenderTarget({ RHIRenderTargetDescriptorElement(vd.m_disparityTexture, level) }));
      }
    }

    // Copy settings to SHM.
    // The worker will read them on the next commit, see the settingsGeneration change, and update itself
    m_depthMapSHM->segment()->m_settingsGeneration += 1;
    m_depthMapSHM->segment()->m_algorithm = m_algorithm;
    m_depthMapSHM->segment()->m_numDisparities = NUM_DISP;
    m_depthMapSHM->segment()->m_useDisparityFilter = m_useDisparityFilter;
    m_depthMapSHM->segment()->m_disparityFilterRadius = m_disparityFilterRadius;
    m_depthMapSHM->segment()->m_disparityFilterIterations = m_disparityFilterIterations;
    m_depthMapSHM->segment()->m_sbmBlockSize = m_sbmBlockSize;
    m_depthMapSHM->segment()->m_sbpIterations = m_sbpIterations;
    m_depthMapSHM->segment()->m_sbpLevels = m_sbpLevels;
    m_depthMapSHM->segment()->m_scsbpNrPlane = m_scsbpNrPlane;
    m_depthMapSHM->segment()->m_sgmP1 = m_sgmP1;
    m_depthMapSHM->segment()->m_sgmP2 = m_sgmP2;
    m_depthMapSHM->segment()->m_sgmUniquenessRatio = m_sgmUniquenessRatio;
    m_depthMapSHM->segment()->m_sgmUseHH4 = m_sgmUseHH4;

    m_didChangeSettings = false;
  }

  // Setup: Copy input camera surfaces to CV GpuMats, resize/grayscale/normalize
  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    ViewData& vd = m_viewData[viewIdx];

    if (!vd.m_isStereoView)
      continue;

    m_cameraSystem->cameraProvider()->populateGpuMat(m_cameraSystem->viewAtIndex(viewIdx).cameraIndices[0], vd.origLeft_gpu/*, m_leftStream*/);
    m_cameraSystem->cameraProvider()->populateGpuMat(m_cameraSystem->viewAtIndex(viewIdx).cameraIndices[1], vd.origRight_gpu/*, m_rightStream*/);
  }

  if (m_enableProfiling) {
    m_setupStartEvent.record(m_globalStream);
  }

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    ViewData& vd = m_viewData[viewIdx];

    if (!vd.m_isStereoView)
      continue;

    cv::cuda::remap(vd.origLeft_gpu, vd.rectLeft_gpu, vd.m_leftMap1_gpu, vd.m_leftMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue=*/ cv::Scalar(), vd.m_leftStream);
    cv::cuda::remap(vd.origRight_gpu, vd.rectRight_gpu, vd.m_rightMap1_gpu, vd.m_rightMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue=*/ cv::Scalar(), vd.m_rightStream);

    cv::cuda::resize(vd.rectLeft_gpu, vd.resizedLeft_gpu, cv::Size(m_iFBAlgoWidth, m_iFBAlgoHeight), 0, 0, cv::INTER_LINEAR, vd.m_leftStream);
    cv::cuda::resize(vd.rectRight_gpu, vd.resizedRight_gpu, cv::Size(m_iFBAlgoWidth, m_iFBAlgoHeight), 0, 0, cv::INTER_LINEAR, vd.m_rightStream);

    cv::cuda::cvtColor(vd.resizedLeft_gpu, vd.resizedLeftGray_gpu, cv::COLOR_BGRA2GRAY, 0, vd.m_leftStream);
    cv::cuda::cvtColor(vd.resizedRight_gpu, vd.resizedRightGray_gpu, cv::COLOR_BGRA2GRAY, 0, vd.m_rightStream);

    if (vd.m_isVerticalStereo) {
      cv::cuda::transpose(vd.resizedLeftGray_gpu, vd.resizedTransposedLeftGray_gpu, vd.m_leftStream);
      cv::cuda::transpose(vd.resizedRightGray_gpu, vd.resizedTransposedRightGray_gpu, vd.m_rightStream);
    }

    vd.m_leftJoinEvent.record(vd.m_leftStream);
    vd.m_rightJoinEvent.record(vd.m_rightStream);
  }

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    ViewData& vd = m_viewData[viewIdx];

    if (!vd.m_isStereoView)
      continue;

    m_globalStream.waitEvent(vd.m_leftJoinEvent);
    m_globalStream.waitEvent(vd.m_rightJoinEvent);
  }


  if (m_enableProfiling) {
    m_setupFinishedEvent.record(m_globalStream);
  }


  // Wait for previous processing to finish
  uint64_t wait_start = currentTimeNs();
  sem_wait(&m_depthMapSHM->segment()->m_workFinishedSem);
  m_syncTimeMs = deltaTimeMs(wait_start, currentTimeNs());
  m_algoTimeMs = m_depthMapSHM->segment()->m_frameTimeMs;



  // Setup view data in the SHM segment
  m_depthMapSHM->segment()->m_activeViewCount = 0;
  size_t lastOffset = 0;

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    ViewData& vd = m_viewData[viewIdx];
    if (!vd.m_isStereoView)
      continue;

    vd.m_shmViewIndex = m_depthMapSHM->segment()->m_activeViewCount;
    m_depthMapSHM->segment()->m_activeViewCount += 1;

    DepthMapSHM::ViewParams& viewParams = m_depthMapSHM->segment()->m_viewParams[vd.m_shmViewIndex];
    cv::cuda::GpuMat& leftGpu = vd.m_isVerticalStereo ? vd.resizedTransposedLeftGray_gpu : vd.resizedLeftGray_gpu;
    cv::cuda::GpuMat& rightGpu = vd.m_isVerticalStereo ? vd.resizedTransposedRightGray_gpu : vd.resizedRightGray_gpu;

    viewParams.width = leftGpu.cols;
    viewParams.height = leftGpu.rows;
    viewParams.inputPitchBytes = leftGpu.step;
    viewParams.outputPitchBytes = viewParams.width * ((m_algorithm == 0) ? 1 : 2); // Tightly packed output so it can be passed to RHI::loadTextureData

    // Allocate space in the SHM region for the I/O buffers.
    size_t inputBufferSize = ((viewParams.height * viewParams.inputPitchBytes) + 4095) & (~4095); // rounded up to pagesize
    size_t outputBufferSize = ((viewParams.height * viewParams.outputPitchBytes) + 4095) & (~4095);

    viewParams.inputLeftOffset = lastOffset;
    viewParams.inputRightOffset = viewParams.inputLeftOffset + inputBufferSize;
    viewParams.outputOffset = viewParams.inputRightOffset + inputBufferSize;

    lastOffset = viewParams.outputOffset + outputBufferSize;

    cv::Mat leftMat(viewParams.height, viewParams.width, CV_8UC1, m_depthMapSHM->segment()->data() + viewParams.inputLeftOffset, viewParams.inputPitchBytes);
    cv::Mat rightMat(viewParams.height, viewParams.width, CV_8UC1, m_depthMapSHM->segment()->data() + viewParams.inputRightOffset, viewParams.inputPitchBytes);
    cv::Mat dispMat(viewParams.height, viewParams.width, (m_algorithm == 0) ? CV_8UC1 : CV_16UC1, m_depthMapSHM->segment()->data() + viewParams.outputOffset, viewParams.outputPitchBytes);

    leftGpu.download(leftMat, m_globalStream);
    rightGpu.download(rightMat, m_globalStream);
  }

  // Write current frame to the SHM segment
  m_globalStream.waitForCompletion(); // finish CUDA->SHM copies

  // Flush all modified regions
  m_depthMapSHM->flush(0, m_depthMapSHM->segment()->m_dataOffset); // header
  for (size_t shmViewIdx = 0; shmViewIdx < m_depthMapSHM->segment()->m_activeViewCount; ++shmViewIdx) {
    DepthMapSHM::ViewParams& viewParams = m_depthMapSHM->segment()->m_viewParams[shmViewIdx];

    m_depthMapSHM->flush(viewParams.inputLeftOffset, viewParams.inputPitchBytes * viewParams.width);
    m_depthMapSHM->flush(viewParams.inputRightOffset, viewParams.inputPitchBytes * viewParams.width);
  }


  // Signal worker to start depth processing
  sem_post(&m_depthMapSHM->segment()->m_workAvailableSem);

  if (m_enableProfiling) {
    m_copyStartEvent.record(m_globalStream);
  }

  // Copy results to render surfaces

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    ViewData& vd = m_viewData[viewIdx];
    if (!vd.m_isStereoView)
      continue;

    DepthMapSHM::ViewParams& viewParams = m_depthMapSHM->segment()->m_viewParams[vd.m_shmViewIndex];

    RHICUDA::copyGpuMatToSurface(vd.rectLeft_gpu, vd.m_iTexture, m_globalStream);

    if (vd.m_isVerticalStereo) {
      cv::Mat transposedDispMat(viewParams.height, viewParams.width, (m_algorithm == 0) ? CV_8UC1 : CV_16UC1, m_depthMapSHM->segment()->data() + viewParams.outputOffset, viewParams.outputPitchBytes);
      cv::Mat dispMat = transposedDispMat.t(); // TODO might be more efficient to transpose in the DGPUWorker while the data is still on the GPU
      rhi()->loadTextureData(vd.m_disparityTexture, (m_algorithm == 0) ? kVertexElementTypeByte1 : kVertexElementTypeShort1, dispMat.data);

    } else {
      rhi()->loadTextureData(vd.m_disparityTexture, (m_algorithm == 0) ? kVertexElementTypeByte1 : kVertexElementTypeShort1, m_depthMapSHM->segment()->data() + viewParams.outputOffset);
    }

    // Filter invalid disparities: generate mip-chain
    for (uint32_t targetLevel = 1; targetLevel < vd.m_disparityTexture->mipLevels(); ++targetLevel) {
      rhi()->beginRenderPass(vd.m_disparityTextureMipTargets[targetLevel], kLoadInvalidate);
      rhi()->bindRenderPipeline(disparityMipPipeline);
      rhi()->loadTexture(ksImageTex, vd.m_disparityTexture);

      DisparityMipUniformBlock ub;
      ub.sourceLevel = targetLevel - 1;

      rhi()->loadUniformBlockImmediate(ksDisparityMipUniformBlock, &ub, sizeof(ub));
      rhi()->drawNDCQuad();
      rhi()->endRenderPass(vd.m_disparityTextureMipTargets[targetLevel]);

    }

    if (m_populateDebugTextures) {
      RHICUDA::copyGpuMatToSurface(vd.resizedLeftGray_gpu, vd.m_leftGray, m_globalStream);
      RHICUDA::copyGpuMatToSurface(vd.resizedRightGray_gpu, vd.m_rightGray, m_globalStream);
    }
  }

  if (m_enableProfiling) {
    m_processingFinishedEvent.record(m_globalStream);
  }

#ifndef GLATTER_EGL_GLES_3_2
  // stupid workaround for profiling on desktop RDMAclient
  m_globalStream.waitForCompletion();
#endif
}

void DepthMapGenerator::renderDisparityDepthMap(size_t viewIdx, const FxRenderView& renderView, const glm::mat4& modelMatrix) {
  ViewData& vd = m_viewData[viewIdx];

  rhi()->bindRenderPipeline(disparityDepthMapPipeline);
  rhi()->bindStreamBuffer(0, m_geoDepthMapTexcoordBuffer);

  MeshDisparityDepthMapUniformBlock ub;
  ub.modelViewProjection = renderView.viewProjectionMatrix * modelMatrix;
  ub.R1inv = vd.m_R1inv;

  ub.Q3 = vd.m_Q[0][3];
  ub.Q7 = vd.m_Q[1][3];
  ub.Q11 = vd.m_Q[2][3];
  ub.CameraDistanceMeters = vd.m_CameraDistanceMeters;
  ub.mogrify = glm::vec2(MOGRIFY_X, MOGRIFY_Y);
  ub.disparityPrescale = m_disparityPrescale;
  ub.disparityTexLevels = vd.m_disparityTexture->mipLevels() - 1;

  ub.trim_minXY = glm::vec2(m_trimLeft, m_trimTop);
  ub.trim_maxXY = glm::vec2((m_iFBAlgoWidth - 1) - m_trimRight, (m_iFBAlgoHeight - 1) - m_trimBottom);

  rhi()->loadUniformBlockImmediate(ksMeshDisparityDepthMapUniformBlock, &ub, sizeof(ub));
  rhi()->loadTexture(ksImageTex, vd.m_iTexture);
  rhi()->loadTexture(ksDisparityTex, vd.m_disparityTexture);

  rhi()->drawIndexedPrimitives(m_geoDepthMapTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapTristripIndexCount);
}

void DepthMapGenerator::renderIMGUI() {
  ImGui::PushID(this);

  ImGui::Text("Stereo Algorithm");
  m_didChangeSettings |= ImGui::RadioButton("BM", &m_algorithm, 0);
  m_didChangeSettings |= ImGui::RadioButton("ConstantSpaceBeliefPropagation", &m_algorithm, 1);
  m_didChangeSettings |= ImGui::RadioButton("SGM", &m_algorithm, 2);

  switch (m_algorithm) {
    case 0: // StereoBM
      m_didChangeSettings |= ImGui::InputInt("Block Size (odd)", &m_sbmBlockSize, /*step=*/2);
      break;
    case 1: // StereoConstantSpaceBP
      m_didChangeSettings |= ImGui::SliderInt("nr_plane", &m_scsbpNrPlane, 1, 16);
      m_didChangeSettings |= ImGui::SliderInt("SBP Iterations", &m_sbpIterations, 1, 8);
      m_didChangeSettings |= ImGui::SliderInt("SBP Levels", &m_sbpLevels, 1, 8);
      break;
    case 2: // StereoSGM
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

  // Render settings. These don't affect the algorithm so we don't need to set m_didChangeSettings when they change.
  ImGui::SliderInt("Trim Left",   &m_trimLeft,   0, 64);
  ImGui::SliderInt("Trim Top",    &m_trimTop,    0, 64);
  ImGui::SliderInt("Trim Right",  &m_trimRight,  0, 64);
  ImGui::SliderInt("Trim Bottom", &m_trimBottom, 0, 64);

  ImGui::Checkbox("CUDA Profiling", &m_enableProfiling);
  if (m_enableProfiling) {
    // TODO use cuEventElapsedTime and skip if the return is not CUDA_SUCCESS --
    // cv::cuda::Event::elapsedTime throws an exception on CUDA_ERROR_NOT_READY
    //float f;
    //if (cuEventElapsedTime(&f, m_setupStartEvent.event
    ImGui::Text("Setup: %.3fms", m_setupTimeMs);
    ImGui::Text("Sync: %.3fms", m_syncTimeMs);
    ImGui::Text("Algo: %.3fms", m_algoTimeMs);
    ImGui::Text("Copy: %.3fms", m_copyTimeMs);
  }

  ImGui::PopID();
}

