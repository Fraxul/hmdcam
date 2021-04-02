#include "imgui.h"
#include "common/DepthMapGenerator.h"
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/RHICVInterop.h"
#include "rhi/gl/GLCommon.h"
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include <thread>

#define NUM_DISP 128 //Max disparity. Must be in {64, 128, 256} for CUDA algorithms.
#define MOGRIFY_X 4
#define MOGRIFY_Y 4

extern RHIRenderPipeline::ptr camTexturedQuadPipeline;
extern FxAtomicString ksNDCQuadUniformBlock;


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

glm::mat4 Matrix4FromCVMatrix(cv::Mat matin) {
  glm::mat4 out(1.0f);
  for (int y = 0; y < matin.rows; y++) {
    for (int x = 0; x < matin.cols; x++) {
      out[y][x] = (float)matin.at<double>(y, x);
    }
  }
  return out;
}

DepthMapGenerator::DepthMapGenerator(CameraSystem* cs, SHMSegment<DepthMapSHM>* shm, size_t viewIdx) :
    m_cameraSystem(cs), m_viewIdx(viewIdx), m_depthMapSHM(shm), m_leftRightJoinEvent(cv::cuda::Event::DISABLE_TIMING), m_enableProfiling(false), m_populateDebugTextures(false) {


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
  // m_disparityTexture = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor(kSurfaceFormat_R16i));
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
        m_disparityPrescale = 1.0f / 16.0f;
        break;
      case 1: {
        if (m_sbpLevels > 4) // more than 4 levels seems to trigger an internal crash
          m_sbpLevels = 4;
        m_disparityPrescale = 1.0f / 16.0f;
      } break;
      case 2:
        m_disparityPrescale = 1.0f / 256.0f; // TODO: not sure if this is correct -- matches results from CSBP, roughly.
        break;
    };

    m_disparityTextureMipTargets.clear();
    if (m_algorithm == 0) {
      // CV_8uc1 disparity map type for StereoBM
      m_disparityTexture = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor::mipDescriptor(kSurfaceFormat_R8i));
    } else {
      // CV_16sc1 disparity map type for everything else
      m_disparityTexture = rhi()->newTexture2D(m_iFBAlgoWidth, m_iFBAlgoHeight, RHISurfaceDescriptor::mipDescriptor(kSurfaceFormat_R16i));
    }

    for (uint32_t level = 0; level < m_disparityTexture->mipLevels(); ++level) {
      m_disparityTextureMipTargets.push_back(rhi()->compileRenderTarget({ RHIRenderTargetDescriptorElement(m_disparityTexture, level) }));
    }

    // Copy settings to SHM.
    // The worker will read them on the next commit and reset m_depthMapSHM->segment()->m_didChangeSettings itself.
    m_depthMapSHM->segment()->m_didChangeSettings = true;
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

#ifdef GLATTER_EGL_GLES_3_2
  // Workaround for rgbTexture() on tegra being an EGLStream-backed texture, which doesn't work correctly with cuda-gl interop mapping
  if (!origLeftBlitRT) {
    origLeftBlitSrf = rhi()->newTexture2D(m_cameraSystem->cameraProvider()->streamWidth(), m_cameraSystem->cameraProvider()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    origRightBlitSrf = rhi()->newTexture2D(m_cameraSystem->cameraProvider()->streamWidth(), m_cameraSystem->cameraProvider()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    origLeftBlitRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor( { origLeftBlitSrf } ));
    origRightBlitRT = rhi()->compileRenderTarget(RHIRenderTargetDescriptor( { origRightBlitSrf } ));
  }

  {
    glm::mat4 ub = glm::mat4(1.0f);
    ub[1][1] = -1.0f; // flip Y for coordsys fix
    {
      rhi()->beginRenderPass(origLeftBlitRT, kLoadInvalidate);
      rhi()->bindRenderPipeline(camTexturedQuadPipeline);
      rhi()->loadTexture(ksImageTex, m_cameraSystem->cameraProvider()->rgbTexture(m_cameraSystem->viewAtIndex(m_viewIdx).cameraIndices[0]), linearClampSampler);
      rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(glm::mat4));
      rhi()->drawNDCQuad();
      rhi()->endRenderPass(origLeftBlitRT);
    }
    {
      rhi()->beginRenderPass(origRightBlitRT, kLoadInvalidate);
      rhi()->bindRenderPipeline(camTexturedQuadPipeline);
      rhi()->loadTexture(ksImageTex, m_cameraSystem->cameraProvider()->rgbTexture(m_cameraSystem->viewAtIndex(m_viewIdx).cameraIndices[1]), linearClampSampler);
      rhi()->loadUniformBlockImmediate(ksNDCQuadUniformBlock, &ub, sizeof(glm::mat4));
      rhi()->drawNDCQuad();
      rhi()->endRenderPass(origRightBlitRT);
    }
  }

  RHICUDA::copySurfaceToGpuMat(origLeftBlitSrf, origLeft_gpu/*, m_leftStream*/);
  RHICUDA::copySurfaceToGpuMat(origRightBlitSrf, origRight_gpu/*, m_rightStream*/);


#else
  RHICUDA::copySurfaceToGpuMat(m_cameraSystem->cameraProvider()->rgbTexture(m_cameraSystem->viewAtIndex(m_viewIdx).cameraIndices[0]), origLeft_gpu/*, m_leftStream*/);
  RHICUDA::copySurfaceToGpuMat(m_cameraSystem->cameraProvider()->rgbTexture(m_cameraSystem->viewAtIndex(m_viewIdx).cameraIndices[1]), origRight_gpu/*, m_rightStream*/);
#endif

  if (m_enableProfiling) {
    m_setupStartEvent.record(m_leftStream);
  }

  cv::cuda::remap(origLeft_gpu, rectLeft_gpu, m_leftMap1_gpu, m_leftMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue=*/ cv::Scalar(), m_leftStream);
  cv::cuda::remap(origRight_gpu, rectRight_gpu, m_rightMap1_gpu, m_rightMap2_gpu, CV_INTER_LINEAR, cv::BORDER_CONSTANT, /*borderValue=*/ cv::Scalar(), m_rightStream);

  cv::cuda::resize(rectLeft_gpu, resizedLeft_gpu, cv::Size(m_iFBAlgoWidth, m_iFBAlgoHeight), 0, 0, cv::INTER_LINEAR, m_leftStream);
  cv::cuda::resize(rectRight_gpu, resizedRight_gpu, cv::Size(m_iFBAlgoWidth, m_iFBAlgoHeight), 0, 0, cv::INTER_LINEAR, m_rightStream);

  cv::cuda::cvtColor(resizedLeft_gpu, resizedLeftGray_gpu, cv::COLOR_BGRA2GRAY, 0, m_leftStream);
  cv::cuda::cvtColor(resizedRight_gpu, resizedRightGray_gpu, cv::COLOR_BGRA2GRAY, 0, m_rightStream);

  m_leftRightJoinEvent.record(m_rightStream);
  m_leftStream.waitEvent(m_leftRightJoinEvent);

  if (m_enableProfiling) {
    m_setupFinishedEvent.record(m_leftStream);
  }


  m_depthMapSHM->segment()->m_activeViewCount = 1; // TODO multiview support
  DepthMapSHM::ViewParams& viewParams = m_depthMapSHM->segment()->m_viewParams[0];
  viewParams.width = resizedLeftGray_gpu.cols;
  viewParams.height = resizedLeftGray_gpu.rows;
  viewParams.inputPitchBytes = resizedLeftGray_gpu.step;
  viewParams.outputPitchBytes = viewParams.width * ((m_algorithm == 0) ? 1 : 2); // Tightly packed output so it can be passed to RHI::loadTextureData
  viewParams.inputLeftOffset = 0;
  viewParams.inputRightOffset = (viewParams.inputLeftOffset + (viewParams.height * viewParams.inputPitchBytes) + 4095) & (~4095);
  viewParams.outputOffset = (viewParams.inputRightOffset + (viewParams.height * viewParams.inputPitchBytes) + 4095) & (~4095);


#if 0
  // DEBUG: process everything twice for perf testing
  memcpy(&(m_depthMapSHM->segment()->m_viewParams[1]), &(m_depthMapSHM->segment()->m_viewParams[0]), sizeof(DepthMapSHM::ViewParams));
  m_depthMapSHM->segment()->m_activeViewCount = 2;
#endif


  cv::Mat leftMat(viewParams.height, viewParams.width, CV_8UC1, m_depthMapSHM->segment()->data() + viewParams.inputLeftOffset, viewParams.inputPitchBytes);
  cv::Mat rightMat(viewParams.height, viewParams.width, CV_8UC1, m_depthMapSHM->segment()->data() + viewParams.inputRightOffset, viewParams.inputPitchBytes);
  cv::Mat dispMat(viewParams.height, viewParams.width, (m_algorithm == 0) ? CV_8UC1 : CV_16UC1, m_depthMapSHM->segment()->data() + viewParams.outputOffset, viewParams.outputPitchBytes);

  // Disparity computation

  // Readback previous results from the SHM segment
 
  uint64_t wait_start = currentTimeNs();
  sem_wait(&m_depthMapSHM->segment()->m_workFinishedSem);
  m_syncTimeMs = deltaTimeMs(wait_start, currentTimeNs());

  m_algoTimeMs = m_depthMapSHM->segment()->m_frameTimeMs;

  // Write current frame to the SHM segment
  m_leftStream.waitForCompletion(); // sync for transfer to other GPU

  resizedLeftGray_gpu.download(leftMat);
  resizedRightGray_gpu.download(rightMat);

  m_depthMapSHM->flush(viewParams.inputLeftOffset, viewParams.inputPitchBytes * viewParams.width);
  m_depthMapSHM->flush(viewParams.inputRightOffset, viewParams.inputPitchBytes * viewParams.width);

  // Signal processing start
  sem_post(&m_depthMapSHM->segment()->m_workAvailableSem);

  //m_depthMapAlgorithm->processFrame(resizedLeftGray_gpu.cols, resizedLeftGray_gpu.rows, resizedLeftGray_gpu.cudaPtr(), resizedRightGray_gpu.cudaPtr(), resizedLeftGray_gpu.step, mdisparity_gpu.cudaPtr(), mdisparity_gpu.step);

  if (m_enableProfiling) {
    m_copyStartEvent.record(m_leftStream);
  }

  // Copy results to render surfaces

  RHICUDA::copyGpuMatToSurface(rectLeft_gpu, m_iTexture, m_leftStream);
  rhi()->loadTextureData(m_disparityTexture, (m_algorithm == 0) ? kVertexElementTypeByte1 : kVertexElementTypeShort1, m_depthMapSHM->segment()->data() + viewParams.outputOffset);

  // Filter invalid disparities: generate mip-chain
  for (uint32_t targetLevel = 1; targetLevel < m_disparityTexture->mipLevels(); ++targetLevel) {
    rhi()->beginRenderPass(m_disparityTextureMipTargets[targetLevel], kLoadInvalidate);
    rhi()->bindRenderPipeline(disparityMipPipeline);
    rhi()->loadTexture(ksImageTex, m_disparityTexture);

    DisparityMipUniformBlock ub;
    ub.sourceLevel = targetLevel - 1;

    rhi()->loadUniformBlockImmediate(ksDisparityMipUniformBlock, &ub, sizeof(ub));
    rhi()->drawNDCQuad();
    rhi()->endRenderPass(m_disparityTextureMipTargets[targetLevel]);

  }

  glFinish(); // XXX: Workaround for mipchain read corruption

  if (m_populateDebugTextures) {
    RHICUDA::copyGpuMatToSurface(resizedLeftGray_gpu, m_leftGray, m_leftStream);
    RHICUDA::copyGpuMatToSurface(resizedRightGray_gpu, m_rightGray, m_leftStream);
  }

  if (m_enableProfiling) {
    m_processingFinishedEvent.record(m_leftStream);
  }

#ifndef GLATTER_EGL_GLES_3_2
  // stupid workaround for profiling on desktop RDMAclient
  m_leftStream.waitForCompletion();
#endif
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
  ub.disparityPrescale = m_disparityPrescale;
  ub.disparityTexLevels = m_disparityTexture->mipLevels() - 1;

  rhi()->loadUniformBlockImmediate(ksMeshDisparityDepthMapUniformBlock, &ub, sizeof(ub));
  rhi()->loadTexture(ksImageTex, m_iTexture);
  rhi()->loadTexture(ksDisparityTex, m_disparityTexture);

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

