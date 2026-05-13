#include "imgui.h"
#include "common/DepthMapGenerator.h"
#include "common/DepthMapGeneratorMock.h"
#ifdef HAVE_OPENCV_CUDA
#include "common/DepthMapGeneratorSHM.h"
#endif
#ifdef L4T_RELEASE_MAJOR
#include "common/tegra/DepthMapGeneratorOFA.h"
#endif
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/Timing.h"
#include "common/glmCvInterop.h"
#include "common/disparityOcclusionMask.h"
#include "common/disparityTemporalFilter.h"
#include "common/medianFilter.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/RHICVInterop.h"
#include "rhi/gl/GLCommon.h"
#include "rhi/vk/RHIInteropBufferGL.h"
#include "rhi/vk/RHIInteropSurfaceGL.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <epoxy/gl.h> // epoxy_is_desktop_gl
#include <glm/gtc/packing.hpp>
#include <npp.h>

const char* settingsFilename = "depthMapSettings.yml";

DepthMapGeneratorBackend depthBackendStringToEnum(const char* backendStr) {
  if (!strcasecmp(backendStr, "none")) {
    return kDepthBackendNone;
  } else if (!strcasecmp(backendStr, "mock")) {
    return kDepthBackendMock;
  } else if ((!strcasecmp(backendStr, "dgpu")) || (!strcasecmp(backendStr, "cuda"))) {
    return kDepthBackendDGPU;
  } else if ((!strcasecmp(backendStr, "depthai")) || (!strcasecmp(backendStr, "depth-ai"))) {
    return kDepthBackendDepthAI;
  } else if (!strcasecmp(backendStr, "ofa")) {
    return kDepthBackendOFA;
  } else {
    fprintf(stderr, "depthBackendStringToEnum: unrecognized worker type \"%s\"\n", backendStr);
    return kDepthBackendNone;
  }
}

DepthMapGenerator* createDepthMapGenerator(DepthMapGeneratorBackend backend) {
  switch (backend) {
    case kDepthBackendNone:
      return NULL;

    case kDepthBackendMock:
      return new DepthMapGeneratorMock();

    case kDepthBackendDGPU:
    case kDepthBackendDepthAI:
#ifdef HAVE_OPENCV_CUDA
      return new DepthMapGeneratorSHM(backend);
#else
      assert(false && "createDepthMapGenerator: SHM-based backends were disabled at compile time (no opencv_cudaimgproc support).");
#endif

    case kDepthBackendOFA:
#ifdef L4T_RELEASE_MAJOR
      return new DepthMapGeneratorOFA();
#else
      assert(false && "createDepthMapGenerator: OFA backend was disabled at compile time (not building on Tegra).");
#endif

    default:
      assert(false && "createDepthMapGenerator: Unhandled backend enum");
  };
  return NULL;
}

FxAtomicString ksMeshDisparityDepthMapUniformBlock("MeshDisparityDepthMapUniformBlock");
static FxAtomicString ksDisparityTex("disparityTex");

static FxAtomicString ksSrcImage("srcImage");
static FxAtomicString ksDstMip1("dstMip1");
static FxAtomicString ksDstMip2("dstMip2");
static FxAtomicString ksDstMip3("dstMip3");

extern FxAtomicString ksDistortionMap;
struct MeshDisparityDepthMapUniformBlock {
  glm::mat4 modelViewProjection[2];
  glm::mat4 R1;
  glm::vec4 depthParameters;

  glm::vec2 mogrify;
  float disparityPrescale;
  int32_t debugFixedDisparity;

  glm::vec2 trim_minXY;
  glm::vec2 trim_maxXY;

  uint32_t renderStereo;
  float maxValidDisparityPixels;
  uint32_t maxValidDisparityRaw;
  float maxDepthDiscontinuity;

  glm::vec2 texCoordStep;
  float minDepthCutoff;
  float pointScale;

  glm::vec2 inputImageSize;
  float pad3, pad4;
};

FxAtomicString ksDisparityMipUniformBlock("DisparityMipUniformBlock");
struct DisparityMipUniformBlock {
  uint32_t sourceLevel;
  uint32_t maxValidDisparityRaw;
  float pad3, pad4;
};

DepthMapGenerator::DepthMapGenerator(DepthMapGeneratorBackend backend_) :
  m_backend(backend_) {
  memset(&m_nppStreamContext, 0, sizeof(m_nppStreamContext));
  NPP_CHECK(nppSetStream((CUstream) m_globalStream.cudaPtr()));
  NPP_CHECK(nppGetStreamContext(&m_nppStreamContext));

  CUDA_CHECK(cuEventCreate(&m_finalizeDisparityStartEvent, CU_EVENT_DEFAULT));
  CUDA_CHECK(cuEventCreate(&m_finalizeDisparityFinishedEvent, CU_EVENT_DEFAULT));

  // Set up a good initial state for the frame timing events
  CUDA_CHECK_NONFATAL(cuEventRecord(m_finalizeDisparityStartEvent, (CUstream) m_globalStream.cudaPtr()));
  CUDA_CHECK_NONFATAL(cuEventRecord(m_finalizeDisparityFinishedEvent, (CUstream) m_globalStream.cudaPtr()));
}

void DepthMapGenerator::initWithCameraSystem(CameraSystem* cs) {
  m_cameraSystem = cs;

  // Create interop sync. initWithCameraSystem should happen late enough that we have RHI access.
  m_interopSync = new RHIInteropSync();

  // Compute internal dimensions
  m_internalWidth = cameraStreamWidth() / m_algoDownsampleX;
  m_internalHeight = cameraStreamHeight() / m_algoDownsampleY;

  // Algorithm input size defaults to the same as internalWidth/internalHeight.
  m_algoInputWidth = m_internalWidth;
  m_algoInputHeight = m_internalHeight;

  // Create depth map geometry buffers
  {
    { // Texcoord and position buffers
      std::vector<float> depth_tc;
      uint32_t uiDepthVertCount = internalWidth() * internalHeight();
      depth_tc.resize(uiDepthVertCount * 4);
      for (uint32_t y = 0; y < internalHeight(); y++) {
        for (uint32_t x = 0; x < internalWidth(); x++) {
          // xy is image texture coordinates (0...1)
          depth_tc[(x + y * internalWidth()) * 4 + 0] = static_cast<float>(x) / static_cast<float>(internalWidth() - 1);
          depth_tc[(x + y * internalWidth()) * 4 + 1] = static_cast<float>(y) / static_cast<float>(internalHeight() - 1);
          // zw is disparity map coordinates (texels)
          depth_tc[(x + y * internalWidth()) * 4 + 2] = x;
          depth_tc[(x + y * internalWidth()) * 4 + 3] = y;
        }
      }
      m_geoDepthMapTexcoordBuffer = rhi()->newBufferWithContents(depth_tc.data(), depth_tc.size() * sizeof(float), kBufferUsageCPUWriteOnly);
    }


    uint32_t dmxm1 = internalWidth() - 1;
    uint32_t dmym1 = internalHeight() - 1;
    { // Tristrip indices
      //From https://github.com/cnlohr/spreadgine/blob/master/src/spreadgine_util.c:216
      std::vector<uint32_t> depth_ia;
      depth_ia.reserve((internalWidth() * dmym1 * 2) + dmym1);
      //uint32_t uiDepthIndexCount = (uint32_t)depth_ia.size();
      for (uint32_t y = 0; y < dmym1; y++) {
        if (y != 0)
          depth_ia.push_back(0xffffffff); // strip-restart

        for (uint32_t x = 0; x < internalWidth(); x++) {
          // clang-format off
          depth_ia.push_back(x + ( y      * (internalWidth())));
          depth_ia.push_back(x + ((y + 1) * (internalWidth())));
          // clang-format on
        }
      }

      m_geoDepthMapTristripIndexBuffer = rhi()->newBufferWithContents(depth_ia.data(), depth_ia.size() * sizeof(uint32_t), kBufferUsageCPUWriteOnly);
      m_geoDepthMapTristripIndexCount = depth_ia.size();
    }

    { // Line indices
      std::vector<uint32_t> depth_ia_lines;
      depth_ia_lines.resize(internalWidth() * dmym1 * 2);
      //uint32_t uiDepthIndexCountLines = (unsigned int)depth_ia_lines.size();

      for (uint32_t y = 0; y < dmym1; y++) {
        for (uint32_t x = 0; x < internalWidth(); x += 2) {
          uint32_t sq = (x + y * dmxm1) * 2;
          // clang-format off
          depth_ia_lines[sq + 0] = x + y * (internalWidth());
          depth_ia_lines[sq + 1] = (x + 1) + (y) * (internalWidth());
          depth_ia_lines[sq + 2] = (x + 1) + (y + 1) * (internalWidth());
          depth_ia_lines[sq + 3] = (x + 2) + (y + 1) * (internalWidth());
          // clang-format on
        }
      }
      m_geoDepthMapLineIndexBuffer = rhi()->newBufferWithContents(depth_ia_lines.data(), depth_ia_lines.size() * sizeof(uint32_t), kBufferUsageCPUWriteOnly);
      m_geoDepthMapLineIndexCount = depth_ia_lines.size();
    }


    { // Point-rendering vertex + index buffer
      std::vector<uint16_t> depth_tc;
      std::vector<uint32_t> depth_ia;
      depth_tc.reserve(internalWidth() * internalHeight() * 4 * 4);
      depth_ia.reserve(internalWidth() * internalHeight() * 5);
      size_t counter = 0;

      for (uint32_t y = 0; y < internalHeight(); y++) {
        for (uint32_t x = 0; x < internalWidth(); x++) {
          // [0] is disparity sample coordinates (integer texels)
          // [1] is offset in current prim (0...1 across the quad)
          // clang-format off
          depth_tc.push_back(x); depth_tc.push_back(y);
          depth_tc.push_back(0); depth_tc.push_back(0);
          depth_ia.push_back(counter++);

          depth_tc.push_back(x); depth_tc.push_back(y);
          depth_tc.push_back(0); depth_tc.push_back(1);
          depth_ia.push_back(counter++);

          depth_tc.push_back(x); depth_tc.push_back(y);
          depth_tc.push_back(1); depth_tc.push_back(0);
          depth_ia.push_back(counter++);

          depth_tc.push_back(x); depth_tc.push_back(y);
          depth_tc.push_back(1); depth_tc.push_back(1);
          depth_ia.push_back(counter++);
          depth_ia.push_back(0xffffffff); // strip-restart
          // clang-format on
        }
      }
      m_geoDepthMapPointTexcoordBuffer = rhi()->newBufferWithContents(depth_tc.data(), depth_tc.size() * sizeof(depth_tc[0]), kBufferUsageCPUWriteOnly);

      depth_ia.pop_back(); // remove the unneccesary last strip-restart index

      m_geoDepthMapPointTristripIndexBuffer = rhi()->newBufferWithContents(depth_ia.data(), depth_ia.size() * sizeof(uint32_t), kBufferUsageCPUWriteOnly);
      m_geoDepthMapPointTristripIndexCount = depth_ia.size();
    }
  }

  {
    // Basic point-rendering pipeline: each disparity texel is mapped to a quad.
    // clang-format off
    RHIRenderPipelineDescriptor rpd;
    rpd.primitiveTopology = kPrimitiveTopologyTriangleStrip;
    rpd.primitiveRestartEnabled = true;

    RHIShaderDescriptor desc("shaders/meshDisparityDepthMapPoints.vtx.glsl", "shaders/meshDisparityDepthMapPoints.frag.glsl", RHIVertexLayout({
        RHIVertexLayoutElement(0, kVertexElementTypeUShort2, "disparitySampleCoordinates", 0, 8),
        RHIVertexLayoutElement(0, kVertexElementTypeUShort2, "quadCoordOffset",            4, 8)
      }));

    desc.setFlag("SAMPLER_TYPE", cs->cameraProvider()->rgbTextureGLSamplerType());

    m_disparityDepthMapPointsPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), rpd);
    // clang-format on
  }

  {
    // Adaptive-mesh pipeline: a triangle list assembled per-frame in CUDA from the
    // post-processed disparity. Each emitted patch is a 2-triangle quad whose corner
    // disparities are sampled at the corner positions.
    // clang-format off
    RHIRenderPipelineDescriptor rpd;
    rpd.primitiveTopology = kPrimitiveTopologyTriangleList;
    rpd.primitiveRestartEnabled = false;

    RHIShaderDescriptor desc("shaders/meshDisparityDepthMapAdaptive.vtx.glsl", "shaders/meshDisparityDepthMapAdaptive.frag.glsl", RHIVertexLayout({
        RHIVertexLayoutElement(0, kVertexElementTypeUShort2, "gridCoord",      offsetof(AdaptiveMeshVertex, gridX),        sizeof(AdaptiveMeshVertex)),
        RHIVertexLayoutElement(0, kVertexElementTypeFloat1,  "disparityRawIn", offsetof(AdaptiveMeshVertex, disparityRaw), sizeof(AdaptiveMeshVertex))
      }));

    desc.setFlag("SAMPLER_TYPE", cs->cameraProvider()->rgbTextureGLSamplerType());

    m_disparityDepthMapAdaptivePipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), rpd);
    // clang-format on
  }

  // Allow derived classes to do additional init after the CameraSystem is known
  this->internalPostInitWithCameraSystem();
}

void DepthMapGenerator::internalPostInitWithCameraSystem() {
  // Empty default implementation
}

#define readNode(node, settingName) cv::read(node[#settingName], m_##settingName, m_##settingName)

bool DepthMapGenerator::loadSettings() {
  cv::FileStorage fs(settingsFilename, cv::FileStorage::READ | cv::FileStorage::FORMAT_YAML);
  if (!fs.isOpened()) {
    printf("DepthMapGenerator: unable to open settings file\n");
    return false;
  }

  try {
    // Load common render settings
    cv::FileNode rsn = fs["renderSettings"];
    if (rsn.isMap()) {
      readNode(rsn, splitDepthDiscontinuity);
      readNode(rsn, maxDepthDiscontinuity);
      readNode(rsn, minDepthCutoff);
      readNode(rsn, usePointRendering);
      readNode(rsn, pointScale);
      readNode(rsn, adaptiveFlatnessThreshold);
      readNode(rsn, adaptiveDepthDiscontinuityThreshold);
      readNode(rsn, trimLeft);
      readNode(rsn, trimTop);
      readNode(rsn, trimRight);
      readNode(rsn, trimBottom);
    }

    // Delegate to impl for algorithm settings
    this->internalLoadSettings(fs);

  } catch (const std::exception& ex) {
    printf("Unable to load depth map settings: %s\n", ex.what());
    return false;
  }
  return true;
}
#undef readNode

#define writeNode(fileStorage, settingName) fileStorage.write(#settingName, m_##settingName)
void DepthMapGenerator::saveSettings() {
  cv::FileStorage fs(settingsFilename, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);

  // Delegate to impl for algorithm settings
  this->internalSaveSettings(fs);

  // Write common render settings
  fs.startWriteStruct(cv::String("renderSettings"), cv::FileNode::MAP, cv::String());
  writeNode(fs, splitDepthDiscontinuity);
  writeNode(fs, maxDepthDiscontinuity);
  writeNode(fs, minDepthCutoff);
  writeNode(fs, usePointRendering);
  writeNode(fs, pointScale);
  writeNode(fs, adaptiveFlatnessThreshold);
  writeNode(fs, adaptiveDepthDiscontinuityThreshold);
  writeNode(fs, trimLeft);
  writeNode(fs, trimTop);
  writeNode(fs, trimRight);
  writeNode(fs, trimBottom);
  fs.endWriteStruct();
}
#undef writeNode

DepthMapGenerator::~DepthMapGenerator() {
  for (ViewData* vd : m_viewData) {
    delete vd; // ensure resources are released
  }
  m_viewData.clear();

  cuEventDestroy(m_finalizeDisparityStartEvent);
  cuEventDestroy(m_finalizeDisparityFinishedEvent);
}

bool DepthMapGenerator::internalRenderSetup(size_t viewIdx, bool stereo, const FxRenderView& renderView0, const FxRenderView& renderView1) {
  ViewData* vd = m_viewData[viewIdx];

  if (vd->m_leftCameraStreamFailed && vd->m_rightCameraStreamFailed)
    return false; // Nothing to render for this view!

  if (m_usePointRendering) {
    rhi()->bindRenderPipeline(m_disparityDepthMapPointsPipeline);
    rhi()->bindStreamBuffer(0, m_geoDepthMapPointTexcoordBuffer);
  } else {
    rhi()->bindRenderPipeline(m_disparityDepthMapAdaptivePipeline);
    rhi()->bindStreamBuffer(0, vd->m_adaptiveVertexBuffer);
  }

  MeshDisparityDepthMapUniformBlock ub;
  const glm::mat4 viewWorldTransform = m_cameraSystem->viewWorldTransform(viewIdx);
  // viewWorldTransform will give us a view whose depth is aligned along +Z. We need to rotate it 180 degrees for our -Z view aligment.
  const glm::mat4 rotationCorrection = glm::scale(glm::vec3(-1.0f, 1.0f, -1.0f)); // 180 degree rotation around Y
  ub.modelViewProjection[0] = renderView0.viewProjectionMatrix * rotationCorrection * viewWorldTransform;
  ub.modelViewProjection[1] = renderView1.viewProjectionMatrix * rotationCorrection * viewWorldTransform;
  ub.R1 = vd->m_R1;

  ub.depthParameters = vd->m_depthParameters;

  ub.mogrify = glm::vec2(m_algoDownsampleX, m_algoDownsampleY);
  ub.disparityPrescale = disparityPrescale() * debugDisparityScale();

  // If a camera stream fails, we won't have a valid disparity map -- fall back to fixed-dispairty mode instead.
  if (vd->anyCameraStreamFailed() || m_debugUseFixedDisparity)
    ub.debugFixedDisparity = m_debugFixedDisparityValue;
  else
    ub.debugFixedDisparity = -1;

  ub.trim_minXY = glm::vec2(m_trimLeft, m_trimTop);
  ub.trim_maxXY = glm::vec2((vd->m_disparityTexture->width() - 1) - m_trimRight, (vd->m_disparityTexture->height() - 1) - m_trimBottom);

  ub.renderStereo = (stereo ? 1 : 0);
  ub.maxValidDisparityPixels = maxDisparityPixels() - 1;
  ub.maxValidDisparityRaw = maxDisparityRaw();
  ub.maxDepthDiscontinuity = m_splitDepthDiscontinuity ? m_maxDepthDiscontinuity : FLT_MAX;

  ub.texCoordStep = glm::vec2(
    1.0f / static_cast<float>(internalWidth()),
    1.0f / static_cast<float>(internalHeight()));

  ub.minDepthCutoff = m_minDepthCutoff;
  ub.pointScale = m_pointScale;

  ub.inputImageSize = glm::vec2(m_cameraSystem->cameraProvider()->streamWidth(), m_cameraSystem->cameraProvider()->streamHeight());

  rhi()->loadUniformBlockImmediate(ksMeshDisparityDepthMapUniformBlock, &ub, sizeof(ub));
  rhi()->loadTexture(ksDisparityTex, vd->m_disparityTexture);

  if (vd->m_leftCameraStreamFailed) {
    // Fallback: If the left camera stream has failed, use fixed-disparity mode and the right camera's image instead.
    rhi()->loadTexture(ksImageTex, m_cameraSystem->cameraProvider()->rgbTexture(vd->m_rightCameraIndex), linearClampSampler);
  } else {
    rhi()->loadTexture(ksImageTex, m_cameraSystem->cameraProvider()->rgbTexture(vd->m_leftCameraIndex), linearClampSampler);
  }
  rhi()->loadTexture(ksDistortionMap, m_cameraSystem->viewAtIndex(viewIdx).stereoDistortionMap[0]);

  return true;
}

void DepthMapGenerator::renderDisparityDepthMapStereo(size_t viewIdx, const FxRenderView& leftRenderView, const FxRenderView& rightRenderView) {
  if (!internalRenderSetup(viewIdx, /*stereo=*/ true, leftRenderView, rightRenderView))
    return;

  ViewData* vd = m_viewData[viewIdx];
  if (m_usePointRendering)
    rhi()->drawIndexedPrimitives(m_geoDepthMapPointTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapPointTristripIndexCount, /*indexOffsetElements=*/ 0, /*instanceCount=*/ 2);
  else
    rhi()->drawIndexedPrimitivesIndirect(vd->m_adaptiveIndexBuffer, kIndexBufferTypeUInt32, vd->m_adaptiveIndirectArgsBuffer, /*indirectCommandCount=*/ 1, /*indirectCommandArrayOffset=*/ 0); // slot 0 = stereo (instanceCount=2)
}

void DepthMapGenerator::renderDisparityDepthMap(size_t viewIdx, const FxRenderView& renderView) {
  if (!internalRenderSetup(viewIdx, /*stereo=*/ false, renderView, renderView))
    return;

  ViewData* vd = m_viewData[viewIdx];
  if (m_usePointRendering)
    rhi()->drawIndexedPrimitives(m_geoDepthMapPointTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapPointTristripIndexCount, /*indexOffsetElements=*/ 0, /*instanceCount=*/ 1);
  else
    rhi()->drawIndexedPrimitivesIndirect(vd->m_adaptiveIndexBuffer, kIndexBufferTypeUInt32, vd->m_adaptiveIndirectArgsBuffer, /*indirectCommandCount=*/ 1, /*indirectCommandArrayOffset=*/ 1); // slot 1 = mono (instanceCount=1)
}

void DepthMapGenerator::renderIMGUI() {
  ImGui::PushID(this);

  this->internalRenderIMGUI();

  // Common processing settings
  ImGui::Checkbox("Median filter", &m_useMedianFilter);
  ImGui::Checkbox("Occlusion mask", &m_useOcclusionMask);
  if (m_useOcclusionMask) {
    const uint32_t windowMin = 0, windowMax = 256;
    ImGui::DragScalar("Search window (px, 0=auto)", ImGuiDataType_U32, &m_occlusionMaskSearchWindow, /*v_speed=*/ 1, &windowMin, &windowMax, "%u", /*flags=*/ 0);
    ImGui::DragFloat("Hysteresis (px)", &m_occlusionMaskHysteresis, /*v_speed=*/ 0.25f, /*v_min=*/ 0.0f, /*v_max=*/ 16.0f, "%.2f");
    const uint8_t confMin = 0, confMax = 255;
    ImGui::DragScalar("Confidence ceiling", ImGuiDataType_U8, &m_occlusionMaskConfidenceCeiling, /*v_speed=*/ 1, &confMin, &confMax, "%u", /*flags=*/ 0);
    ImGui::Checkbox("Smear background into deadzone", &m_occlusionMaskSmear);
  }

  ImGui::Checkbox("FGS filter", &m_useFGSFilter);
  if (m_useFGSFilter) {
    ImGui::DragFloat("FGS Lambda", &m_fgsLambda, /*v_speed=*/ 250.0f, /*v_min=*/ 1000.0f, /*v_max=*/ 16000.0f, "%.1f");
    ImGui::DragFloat("FGS Sigma Color", &m_fgsSigmaColor, /*v_speed=*/ 0.125f, /*v_min=*/ 0.0f, /*v_max=*/ 8.0f, "%.3f");
    const uint8_t iterationsMin = 1, iterationsMax = 8;
    ImGui::DragScalar("FGS iterations", ImGuiDataType_U8, &m_fgsIterations, /*v_speed=*/ 1, &iterationsMin, &iterationsMax, "%u");
    ImGui::SliderFloat("FGS Lambda Attenuation", &m_fgsLambdaAttenuation, 0.0f, 1.0f);
  }

  ImGui::Checkbox("Temporal filter", &m_useTemporalFilter);

  if (m_useTemporalFilter) {
    const uint8_t alphaMin = 0, alphaMax = 255;
    ImGui::DragScalar("Temporal Filter Alpha", ImGuiDataType_U8, &m_temporalFilterAlpha, /*v_speed=*/ 16, &alphaMin, &alphaMax, "%u", /*flags=*/ 0);

    ImGui::DragFloat("Stable Threshold", &m_temporalFilterStableThreshold, /*v_speed=*/ 0.25f, /*v_min=*/ 0.0f, /*v_max=*/ 32.0f, "%.2fpx", /*flags=*/ 0);
  }

  // Common render settings -- these don't affect the algorithm.
  ImGui::Checkbox("Split depth discontinuity", &m_splitDepthDiscontinuity);
  if (m_splitDepthDiscontinuity)
    ImGui::SliderFloat("Depth Discontinuity", &m_maxDepthDiscontinuity, 0.01f, 2.0f);

  // clang-format off
  ImGui::SliderInt("Trim Left",   &m_trimLeft,   0, 64);
  ImGui::SliderInt("Trim Top",    &m_trimTop,    0, 64);
  ImGui::SliderInt("Trim Right",  &m_trimRight,  0, 64);
  ImGui::SliderInt("Trim Bottom", &m_trimBottom, 0, 64);
  // clang-format on

  ImGui::SliderFloat("Min Depth Cutoff", &m_minDepthCutoff, 0.01f, 0.30f);

  ImGui::Checkbox("Point rendering", &m_usePointRendering);
  if (m_usePointRendering) {
    ImGui::SliderFloat("Point Scale", &m_pointScale, 0.5f, 3.0f);
  } else {
    ImGui::DragInt("Adaptive flatness threshold (raw)", &m_adaptiveFlatnessThreshold, /*v_speed=*/ 8, 0, 128);
    ImGui::DragInt("Adaptive discontinuity threshold (raw)", &m_adaptiveDepthDiscontinuityThreshold, /*v_speed=*/ 8, 0, 1024);
    ImGui::SliderFloat("Adaptive cell overlap multiplier", &m_adaptiveCellOverlapMultiplier, 1.0f, 3.0f, "%.2f");

    // Per-view emission histogram for the most-recent build. The host-side mirror is
    // populated by an async DtoH copy at the end of the build.
    // We don't sync the stream to avoid stalling, and accept the possibility of a torn read.
    // (it's just a debug display, no big deal if the data is a little stale.)
    if (!m_viewData.empty()) {
      for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
        ViewData* vd = m_viewData[viewIdx];
        if (!vd || !vd->m_isStereoView)
          continue;

        const auto& s = vd->m_adaptiveScratch;
        uint32_t totalQuads = s.h_counters->vertexCounter / 4;
        ImGui::Text("View %zu: %u quads", viewIdx, totalQuads);
        for (int L = 0; L < kAdaptiveMeshLevels; ++L) {
          int sz = 1 << L;
          ImGui::Text("  L%d (%dx%d): %u (%.1f%%)", L, sz, sz, s.h_counters->levelHistograms[L],
            totalQuads ? 100.0f * float(s.h_counters->levelHistograms[L]) / float(totalQuads) : 0.0f);
        }
      }
    }
  }

  ImGui::Checkbox("Debug: Fixed disparity", &m_debugUseFixedDisparity);
  if (m_debugUseFixedDisparity)
    ImGui::SliderInt("Fixed Disparity", &m_debugFixedDisparityValue, 0, 256);

  ImGui::PopID();
}


void DepthMapGenerator::renderIMGUIPerformanceGraphs() {
  ImGui::PushID(this);
  this->internalRenderIMGUIPerformanceGraphs();

  ImGui::Text("Finalize: %.1fms", m_finalizeDisparityTimeMs);
  ImGui::PopID();
}

void DepthMapGenerator::processFrame() {

  // Update view data
  if (m_viewData.empty() || (m_viewDataRevision != m_cameraSystem->calibrationDataRevision())) {
    uint64_t startTimeNs = currentTimeNs();

    if (m_viewData.size() > m_cameraSystem->views()) {
      // Trim array
      for (size_t i = m_viewData.size(); i < m_cameraSystem->views(); ++i) {
        delete m_viewData[i];
      }
    }

    m_viewData.resize(m_cameraSystem->views());

    for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
      if (m_viewData[viewIdx] == nullptr)
        m_viewData[viewIdx] = this->newEmptyViewData();


      CameraSystem::View& v = m_cameraSystem->viewAtIndex(viewIdx);
      ViewData* vd = m_viewData[viewIdx];

      vd->m_isStereoView = v.isStereo;
      if (!vd->m_isStereoView)
        continue; // Not applicable for mono views

      vd->m_isVerticalStereo = v.isVerticalStereo();
      vd->m_leftCameraIndex = v.cameraIndices[0];
      vd->m_rightCameraIndex = v.cameraIndices[1];

      vd->m_R1 = glmMat4FromCVMatrix(v.stereoRectification[0]);

      vd->m_depthParameters = v.depthParameters();

      printf("DepthMapGenerator: View %zu stereoRectify depth parameters: [%.3f, %.3f, %.3f, %.3f]\n", viewIdx,
        vd->m_depthParameters[0], vd->m_depthParameters[1], vd->m_depthParameters[2], vd->m_depthParameters[3]);
    }

    // Let the backend impl update its derived view data components
    this->internalUpdateViewData();

    // Rebuild rectified-luma CUtexObjects to match the (possibly reallocated) underlying GpuMats.
    // Backends are only allowed to (re)allocate ViewData buffers inside internalUpdateViewData(),
    // so this is the one place we need to refresh the wrappers.
    for (ViewData* vd : m_viewData) {
      if (vd)
        vd->rebuildRectifiedLumaTextures();
    }

    m_viewDataRevision = m_cameraSystem->calibrationDataRevision();
    uint64_t endTimeNs = currentTimeNs();
    printf("DepthMapGenerator: viewData update took %.3f ms\n", deltaTimeMs(startTimeNs, endTimeNs));
  }

  // Various housekeeping tasks that must happen every frame
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    ViewData* vd = m_viewData[viewIdx];

    // Update stream failure flags
    vd->m_leftCameraStreamFailed = m_cameraSystem->cameraProvider()->isStreamFailed(vd->m_leftCameraIndex);
    vd->m_rightCameraStreamFailed = m_cameraSystem->cameraProvider()->isStreamFailed(vd->m_rightCameraIndex);

    // Swap double-buffered mats
    vd->swapCurrentAndPreviousDisparity();
  }

  this->internalProcessFrame();

  // After frame processing has finished, sync the CUDA updates to GL.
  m_interopSync->signalCUDAToRHI((CUstream) m_globalStream.cudaPtr());
}

void DepthMapGenerator::internalFinalizeDisparityTexture() {

  // Collect profiling data from previous frame
  cuEventElapsedTime(&m_finalizeDisparityTimeMs, m_finalizeDisparityStartEvent, m_finalizeDisparityFinishedEvent);

  CUDA_CHECK(cuEventRecord(m_finalizeDisparityStartEvent, (CUstream) m_globalStream.cudaPtr()));

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView || vd->anyCameraStreamFailed())
      continue;


    cv::cuda::GpuMat* workMat = &vd->currentDisparityMat();

    if (m_useMedianFilter) {
      // Run 3x3 median filter to smooth speckles and edge discontinuities

      medianFilter3x3_u16(*workMat, vd->m_disparityMedianFilterDestGpuMat, (CUstream) m_globalStream.cudaPtr());

      // Work on the median filter destination mat
      workMat = &vd->m_disparityMedianFilterDestGpuMat;
    }

    if (m_useOcclusionMask) {
      // Zero confidence on pixels in the left-of-foreground occlusion shadow
      // so downstream filters treat them as gaps to inpaint rather than as
      // noisy data to average over.
      uint32_t searchWindow = m_occlusionMaskSearchWindow;
      if (searchWindow == 0) {
        // Auto: cover the worst-case shadow width = max disparity in pixels.
        searchWindow = static_cast<uint32_t>(
                         static_cast<float>(maxDisparityRaw()) * disparityPrescale()) +
          1;
      }
      disparityOcclusionMask(
        *workMat, vd->m_disparityConfidence,
        static_cast<uint16_t>(maxDisparityRaw()),
        disparityPrescale(),
        searchWindow,
        m_occlusionMaskHysteresis,
        m_occlusionMaskConfidenceCeiling,
        /*smearLeftScanPixels=*/ m_occlusionMaskSmear ? searchWindow : 0,
        (CUstream) m_globalStream.cudaPtr());
    }

    if (m_useFGSFilter) {
      // Run FGS filter over disparity+confidence

      // Pack (disp*conf, conf) into a CV_32FC2 in one kernel pass.
      fgsPackDispConfMul(*workMat, vd->m_disparityConfidence, disparityPrescale(), vd->m_fgsFilterInOutPacked, (CUstream) m_globalStream.cudaPtr());

      // Fused two-channel filter. Works in-place on vd->m_fgsFilterInOutPacked.
      fgsFilter(vd->m_fgsFilterState,
        vd->m_rectifiedLumaTex[0],
        vd->m_rectifiedLuma[0].cols / workMat->cols, // guide-texture scale; should typically be 1 or 2
        vd->m_fgsFilterInOutPacked,
        m_fgsLambda,
        m_fgsSigmaColor / 255.0f, // guide texture is in integer mode, so sigma does not require the divide-by-255
        /*lambda_attenuation=*/ m_fgsLambdaAttenuation,
        /*num_iter=*/ m_fgsIterations,
        (CUstream) m_globalStream.cudaPtr());
      // Recover the filtered disparity: pair.x / (pair.y + EPS).
      fgsUnpackDivideScale(vd->m_fgsFilterInOutPacked, *workMat, static_cast<float>(1 << m_disparitySubpixelBits), (CUstream) m_globalStream.cudaPtr());
    }

    if (m_useTemporalFilter) {
      // Run temporal filter between current and previous disparity.
      // This always writes to currentDisparityMat, but reads from workMat to avoid a copy
      // in the case where we ran the median filter earlier (which cannot work in-place).
      // (it's OK if workMat and currentDisparityMat are the same mat)

      uint16_t stableThresholdRaw = static_cast<uint16_t>(m_temporalFilterStableThreshold * static_cast<float>(1 << m_disparitySubpixelBits));
      disparityTemporalFilter(maxDisparityRaw(), /*stableDepthThreshold=*/ stableThresholdRaw, /*defaultAlpha*/ m_temporalFilterAlpha,
        /*currentFrameInput=*/ *workMat, /*previousFrameInput=*/ vd->previousDisparityMat(),
        /*output=*/ vd->currentDisparityMat(), (CUstream) m_globalStream.cudaPtr(),
        /*debugMat=*/ nullptr); // &vd->m_disparityDebugResidual);

      // back to reading from currentDisparityMat, since the temporal filter always writes to that.
      workMat = &vd->currentDisparityMat();
    }

    // Copy filtered disparity into the interop surface's CUDA-side storage.
    vd->m_disparityTexture->copyFromGpuMatAsync(*workMat, (CUstream) m_globalStream.cudaPtr());

    // Build the adaptive triangle mesh from the post-processed disparity.
    {
      // Disable trim/validity masking when running with a fixed disparity, so the resulting
      // mesh covers the full grid (matches the legacy fixed-disparity rendering behavior).
      const bool useFixedDisparity = vd->anyCameraStreamFailed() || m_debugUseFixedDisparity;
      const int effTrimL = useFixedDisparity ? 0 : m_trimLeft;
      const int effTrimT = useFixedDisparity ? 0 : m_trimTop;
      const int effTrimR = useFixedDisparity ? 0 : m_trimRight;
      const int effTrimB = useFixedDisparity ? 0 : m_trimBottom;

      buildAdaptiveDepthMesh(
        *workMat,
        static_cast<uint16_t>(maxDisparityRaw()),
        static_cast<uint16_t>(m_adaptiveFlatnessThreshold),
        static_cast<uint16_t>(m_adaptiveDepthDiscontinuityThreshold),
        m_adaptiveCellOverlapMultiplier,
        effTrimL, effTrimT, effTrimR, effTrimB,
        vd->m_adaptiveVertexBuffer->cudaPointer(),
        vd->m_adaptiveIndexBuffer->cudaPointer(),
        vd->m_adaptiveIndirectArgsBuffer->cudaPointer(),
        vd->m_adaptiveScratch,
        (CUstream) m_globalStream.cudaPtr());
    }

    // Populate debug-residual texture.
    if (m_populateDebugTextures) {
      cudaStream_t cudaStream = (cudaStream_t) m_globalStream.cudaPtr();

      if (!vd->m_debugResidual)
        vd->m_debugResidual = RHIInteropSurfaceGL::newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8), RHIInteropSyncDescriptor(m_interopSync, kSyncDirectionCUDAWriter));

      if (!vd->m_confidenceTexture)
        vd->m_confidenceTexture = RHIInteropSurfaceGL::newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8), RHIInteropSyncDescriptor(m_interopSync, kSyncDirectionCUDAWriter));

      vd->m_debugResidual->copyFromGpuMatAsync(vd->m_disparityDebugResidual, cudaStream);

      vd->m_confidenceTexture->copyFromGpuMatAsync(vd->m_disparityConfidence, cudaStream);
    }

    if (debugDisparityCPUAccessEnabled()) {
      // Copy filtered disparity to CPU-visible view
      workMat->download(vd->m_debugCPUDisparity, m_globalStream);
      // Copy confidence to CPU-visible view
      vd->m_disparityConfidence.download(vd->m_debugCPUConfidence, m_globalStream);
      // Copy residual to CPU-visible view
      vd->m_disparityDebugResidual.download(vd->m_debugCPUDisparityResidual, m_globalStream);
    }
  }

  CUDA_CHECK(cuEventRecord(m_finalizeDisparityFinishedEvent, (CUstream) m_globalStream.cudaPtr()));
}


uint32_t divUp(uint32_t x, uint32_t y) {
  return (x + (y - 1)) / y;
}

DepthMapGenerator::ViewData::~ViewData() {
  for (size_t i = 0; i < 2; ++i) {
    if (m_rectifiedLumaTex[i]) {
      cuTexObjectDestroy(m_rectifiedLumaTex[i]);
      m_rectifiedLumaTex[i] = 0;
    }
  }
}

void DepthMapGenerator::ViewData::rebuildRectifiedLumaTextures() {
  for (size_t i = 0; i < 2; ++i) {
    if (m_rectifiedLumaTex[i]) {
      CUDA_CHECK(cuTexObjectDestroy(m_rectifiedLumaTex[i]));
      m_rectifiedLumaTex[i] = 0;
    }
    if (m_rectifiedLuma[i].empty())
      continue;

    assert(m_rectifiedLuma[i].type() == CV_8U);

    CUDA_RESOURCE_DESC resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = CU_RESOURCE_TYPE_PITCH2D;
    resDesc.res.pitch2D.devPtr = (CUdeviceptr) m_rectifiedLuma[i].cudaPtr();
    resDesc.res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT8;
    resDesc.res.pitch2D.numChannels = 1;
    resDesc.res.pitch2D.width = m_rectifiedLuma[i].cols;
    resDesc.res.pitch2D.height = m_rectifiedLuma[i].rows;
    resDesc.res.pitch2D.pitchInBytes = m_rectifiedLuma[i].step;

    CUDA_TEXTURE_DESC texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
    texDesc.maxAnisotropy = 1;

    CUDA_CHECK(cuTexObjectCreate(&m_rectifiedLumaTex[i], &resDesc, &texDesc, /*resourceViewDescriptor=*/ nullptr));
  }
}

void DepthMapGenerator::ViewData::updateDisparityTexture(DepthMapGenerator* depthMapGenerator, uint32_t w, uint32_t h, RHISurfaceFormat format) {
  int cvType = 0;

  switch (format) {
    case kSurfaceFormat_R8i:
      cvType = CV_8U;
      break;

    case kSurfaceFormat_R16i:
      cvType = CV_16U;
      break;
    default:
      assert(false && "updateDisparityTexture: unhandled RHISurfaceFormat");
  };

  // Create disparity mats
  m_disparityGpuMat[0].create(/*rows=*/ h, /*cols=*/ w, /*type=*/ cvType);
  m_disparityGpuMat[1].create(/*rows=*/ h, /*cols=*/ w, /*type=*/ cvType);
  m_disparityConfidence.create(/*rows=*/ h, /*cols=*/ w, /*type=*/ CV_8U);

  // Pre-allocate CPU debug view, identical in size/format to GPU copy
  m_debugCPUDisparity.create(/*rows=*/ h, /*cols=*/ w, /*type=*/ cvType);
  m_debugCPUConfidence.create(/*rows=*/ h, /*cols=*/ w, /*type=*/ CV_8U);

  // Pre-allocate GPU and CPU debug residuals. Always CV_8U.
  m_disparityDebugResidual.create(/*rows=*/ h, /*cols=*/ w, CV_8U);
  m_debugCPUDisparityResidual.create(/*rows=*/ h, /*cols=*/ w, /*type=*/ CV_8U);

  // Pre-allocate CPU debug view of L/R inputs
  for (size_t i = 0; i < 2; ++i) {
    m_debugCPUDisparityInput[i].create(h, w, CV_8U);
  }

  // Allocate buffer for median filter output. This should be identical to m_disparityGpuMat.
  m_disparityMedianFilterDestGpuMat.create(/*rows=*/ h, /*cols=*/ w, /*type=*/ cvType);

  // Pre-allocate FGS filter state
  m_fgsFilterState.ensureAllocated(/*newWidth=*/ w, /*newHeight=*/ h);
  m_fgsFilterInOutPacked.create(/*rows=*/ h, /*cols=*/ w, /*type=*/ CV_32FC2);

  // Disparity output: VK-allocated, dual-imported into GL (texture) and CUDA
  // (the surface's GpuMat). Dropping the legacy newTexture2D + per-frame
  // copyGpuMatToSurface(cuGraphicsMap/Unmap) pair eliminates the ~150-200µs
  // GL-context-switch bubble that used to show up around the map call.
  m_disparityTexture = RHIInteropSurfaceGL::newTexture2D(w, h, RHISurfaceDescriptor(format), RHIInteropSyncDescriptor(depthMapGenerator->m_interopSync, kSyncDirectionCUDAWriter));

  // Adaptive-mesh buffers. Worst case is one quad per leaf cell (every cell stays at level 0).
  // Interop-backed so the CUDA mesh-build kernels can write through a stable
  // CUdeviceptr without a per-frame cuGraphicsMap/Unmap round-trip.
  const size_t worstCaseQuads = size_t(w) * size_t(h);
  const size_t vboBytes = worstCaseQuads * 4 * sizeof(AdaptiveMeshVertex);
  const size_t iboBytes = worstCaseQuads * 6 * sizeof(uint32_t);
  m_adaptiveVertexBuffer = RHIInteropBufferGL::newBuffer(vboBytes, kBufferUsageGPUPrivate, RHIInteropSyncDescriptor(depthMapGenerator->m_interopSync, kSyncDirectionCUDAWriter));
  m_adaptiveIndexBuffer = RHIInteropBufferGL::newBuffer(iboBytes, kBufferUsageGPUPrivate, RHIInteropSyncDescriptor(depthMapGenerator->m_interopSync, kSyncDirectionCUDAWriter));
  m_adaptiveIndirectArgsBuffer = RHIInteropBufferGL::newBuffer(2 * sizeof(DrawElementsIndirectCommand), kBufferUsageGPUPrivate, RHIInteropSyncDescriptor(depthMapGenerator->m_interopSync, kSyncDirectionCUDAWriter));

  m_adaptiveScratch.allocate(w, h);
}

float DepthMapGenerator::debugPeekDisparityTexel(size_t viewIdx, glm::ivec2 texelCoord) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);

  if (vd->m_debugCPUDisparity.empty()) {
    return -1.0f;
  }

  texelCoord = glm::clamp(texelCoord, glm::ivec2(0, 0), glm::ivec2(vd->m_debugCPUDisparity.cols - 1, vd->m_debugCPUDisparity.rows - 1));
  float disparityRaw = 0;

  // .at(row, col) -- Y rows, X columns.
  // clang-format off
  switch (vd->m_debugCPUDisparity.type()) {
    case CV_8U:  disparityRaw = static_cast<float>(vd->m_debugCPUDisparity.at<uint8_t >(texelCoord.y, texelCoord.x)); break;
    case CV_16U: disparityRaw = static_cast<float>(vd->m_debugCPUDisparity.at<uint16_t>(texelCoord.y, texelCoord.x)); break;
    default:
      assert(false && "DepthMapGenerator::debugPeekDisparity: unhandled m_debugCPUDisparity.type()");
  }
  // clang-format on
  return disparityRaw * disparityPrescale();
}

float DepthMapGenerator::debugPeekDisparityUV(size_t viewIdx, glm::vec2 uv) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);
  return debugPeekDisparityTexel(viewIdx, glm::ivec2(uv * (vd->m_disparityTexture->dimensions() - glm::vec2(1.0f, 1.0f))));
}

glm::vec3 DepthMapGenerator::debugPeekLocalPositionUV(size_t viewIdx, glm::vec2 uv) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);
  return debugPeekLocalPositionTexel(viewIdx, glm::ivec2(uv * (vd->m_disparityTexture->dimensions() - glm::vec2(1.0f, 1.0f))));
}

glm::vec3 DepthMapGenerator::debugPeekLocalPositionTexel(size_t viewIdx, glm::ivec2 texelCoord) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);
  float fDisp = debugPeekDisparityTexel(viewIdx, texelCoord);

  glm::vec3 pp = glm::vec3(
    (texelCoord.x * m_algoDownsampleX) + vd->m_depthParameters.x,
    (texelCoord.y * m_algoDownsampleY) + vd->m_depthParameters.y,
    vd->m_depthParameters.z);

  float lw = vd->m_depthParameters.w * (fDisp * m_algoDownsampleX);

  return vd->m_R1 * (pp / lw);
}

float DepthMapGenerator::debugComputeDepthForDisparity(size_t viewIdx, float disparityPixels) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);
  float lz = vd->m_depthParameters[2] / (vd->m_depthParameters[3] * disparityPixels * m_algoDownsampleX);
  return lz;
}

float DepthMapGenerator::debugPeekConfidenceTexel(size_t viewIdx, glm::ivec2 texelCoord) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);

  if (vd->m_debugCPUConfidence.empty()) {
    return 0;
  }
  assert(vd->m_debugCPUConfidence.type() == CV_8U);

  texelCoord = glm::clamp(texelCoord, glm::ivec2(0, 0), glm::ivec2(vd->m_debugCPUConfidence.cols - 1, vd->m_debugCPUConfidence.rows - 1));
  return static_cast<float>(vd->m_debugCPUConfidence.at<uint8_t>(texelCoord.y, texelCoord.x)) / 255.0f;
}

float DepthMapGenerator::debugPeekConfidenceUV(size_t viewIdx, glm::vec2 uv) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);
  return debugPeekConfidenceTexel(viewIdx, glm::ivec2(uv * (vd->m_confidenceTexture->dimensions() - glm::vec2(1.0f, 1.0f))));
}

uint8_t DepthMapGenerator::debugPeekResidualTexel(size_t viewIdx, glm::ivec2 texelCoord) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);

  if (vd->m_debugCPUDisparityResidual.empty()) {
    return 0;
  }
  assert(vd->m_debugCPUDisparityResidual.type() == CV_8U);

  texelCoord = glm::clamp(texelCoord, glm::ivec2(0, 0), glm::ivec2(vd->m_debugCPUDisparityResidual.cols - 1, vd->m_debugCPUDisparityResidual.rows - 1));
  return vd->m_debugCPUDisparityResidual.at<uint8_t>(texelCoord.y, texelCoord.x);
}

uint8_t DepthMapGenerator::debugPeekResidualUV(size_t viewIdx, glm::vec2 uv) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);
  return debugPeekResidualTexel(viewIdx, glm::ivec2(uv * (vd->m_debugResidual->dimensions() - glm::vec2(1.0f, 1.0f))));
}
