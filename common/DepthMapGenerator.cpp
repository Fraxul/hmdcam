#include "imgui.h"
#include "common/DepthMapGenerator.h"
#ifdef HAVE_OPENCV_CUDA
  #include "common/DepthMapGeneratorSHM.h"
#endif
#ifdef HAVE_VPI2
  #include "common/DepthMapGeneratorVPI.h"
#endif
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/Timing.h"
#include "common/glmCvInterop.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/gl/GLCommon.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <epoxy/gl.h> // epoxy_is_desktop_gl

const char* settingsFilename = "depthMapSettings.yml";

DepthMapGeneratorBackend depthBackendStringToEnum(const char* backendStr) {
  if (!strcasecmp(backendStr, "none")) {
    return kDepthBackendNone;
  } else if ((!strcasecmp(backendStr, "dgpu")) || (!strcasecmp(backendStr, "cuda"))) {
    return kDepthBackendDGPU;
  } else if ((!strcasecmp(backendStr, "depthai")) || (!strcasecmp(backendStr, "depth-ai"))) {
    return kDepthBackendDepthAI;
  } else if ((!strcasecmp(backendStr, "vpi")) || (!strcasecmp(backendStr, "vpi2"))) {
    return kDepthBackendVPI;
  } else {
    fprintf(stderr, "depthBackendStringToEnum: unrecognized worker type \"%s\"\n", backendStr);
    return kDepthBackendNone;
  }
}

DepthMapGenerator* createDepthMapGenerator(DepthMapGeneratorBackend backend) {
  switch (backend) {
  case kDepthBackendNone:
    return NULL;

  case kDepthBackendDGPU:
  case kDepthBackendDepthAI:
#ifdef HAVE_OPENCV_CUDA
    return new DepthMapGeneratorSHM(backend);
#else
    assert(false && "createDepthMapGenerator: SHM-based backends were disabled at compile time (no opencv_cudaimgproc support).");
#endif

  case kDepthBackendVPI:
#ifdef HAVE_VPI2
    return new DepthMapGeneratorVPI();
#else
    assert(false && "createDepthMapGenerator: The VPI backend was disabled at compile time.");
#endif

  default:
    assert(false && "createDepthMapGenerator: Unhandled backend enum");
  };
  return NULL;
}

FxAtomicString ksMeshDisparityDepthMapUniformBlock("MeshDisparityDepthMapUniformBlock");
static FxAtomicString ksDisparityTex("disparityTex");
static FxAtomicString ksImageTex("imageTex");

static FxAtomicString ksSrcImage("srcImage");
static FxAtomicString ksDstMip1("dstMip1");
static FxAtomicString ksDstMip2("dstMip2");
static FxAtomicString ksDstMip3("dstMip3");

extern FxAtomicString ksDistortionMap;
struct MeshDisparityDepthMapUniformBlock {
  glm::mat4 modelViewProjection[2];
  glm::mat4 R1inv;
  float Q3, Q7, Q11;
  float CameraDistanceMeters;

  glm::vec2 mogrify;
  float disparityPrescale;
  uint32_t disparityTexLevels;

  glm::vec2 trim_minXY;
  glm::vec2 trim_maxXY;

  uint32_t renderStereo;
  float maxValidDisparityPixels;
  uint32_t maxValidDisparityRaw;
  float maxDepthDiscontinuity;

  glm::vec2 texCoordStep;
  float minDepthCutoff;
  float pointScale;

  int32_t debugFixedDisparity;
  float pad2, pad3, pad4;
};

RHIComputePipeline::ptr disparityMipComputePipeline;
FxAtomicString ksDisparityMipUniformBlock("DisparityMipUniformBlock");
struct DisparityMipUniformBlock {
  uint32_t sourceLevel;
  uint32_t maxValidDisparityRaw;
  float pad3, pad4;
};

DepthMapGenerator::DepthMapGenerator(DepthMapGeneratorBackend backend_) : m_backend(backend_) {

}

void DepthMapGenerator::initWithCameraSystem(CameraSystem* cs) {
  m_cameraSystem = cs;

  // Compute internal dimensions
  m_internalWidth = inputWidth() / m_algoDownsampleX;
  m_internalHeight = inputHeight() / m_algoDownsampleY;

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
          depth_ia.push_back(x + ( y      * (internalWidth())));
          depth_ia.push_back(x + ((y + 1) * (internalWidth())));
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
          depth_ia_lines[sq + 0] = x + y * (internalWidth());
          depth_ia_lines[sq + 1] = (x + 1) + (y) * (internalWidth());
          depth_ia_lines[sq + 2] = (x + 1) + (y + 1) * (internalWidth());
          depth_ia_lines[sq + 3] = (x + 2) + (y + 1) * (internalWidth());
        }
      }
      m_geoDepthMapLineIndexBuffer = rhi()->newBufferWithContents(depth_ia_lines.data(), depth_ia_lines.size() * sizeof(uint32_t), kBufferUsageCPUWriteOnly);
      m_geoDepthMapLineIndexCount = depth_ia_lines.size();
    }


    { // Point-rendering vertex + index buffer
      std::vector<glm::vec2> depth_tc;
      std::vector<uint32_t> depth_ia;
      depth_tc.reserve(internalWidth() * internalHeight() * 3 * 4);
      depth_ia.reserve(internalWidth() * internalHeight() * 5);
      size_t counter = 0;

      for (uint32_t y = 0; y < internalHeight(); y++) {
        for (uint32_t x = 0; x < internalWidth(); x++) {
          // [0] is image texture coordinates (0...1)
          // [1] is disparity sample coordinates (integer texels)
          // [2] is offset in current prim (0...1 across the quad)
          depth_tc.push_back(glm::vec2(static_cast<float>(x    ) / static_cast<float>(internalWidth()), static_cast<float>(y    ) / static_cast<float>(internalHeight())));
          depth_tc.push_back(glm::vec2(x, y));
          depth_tc.push_back(glm::vec2(0, 0));
          depth_ia.push_back(counter++);

          depth_tc.push_back(glm::vec2(static_cast<float>(x    ) / static_cast<float>(internalWidth()), static_cast<float>(y    ) / static_cast<float>(internalHeight())));
          depth_tc.push_back(glm::vec2(x, y));
          depth_tc.push_back(glm::vec2(0, 1));
          depth_ia.push_back(counter++);

          depth_tc.push_back(glm::vec2(static_cast<float>(x    ) / static_cast<float>(internalWidth()), static_cast<float>(y    ) / static_cast<float>(internalHeight())));
          depth_tc.push_back(glm::vec2(x, y));
          depth_tc.push_back(glm::vec2(1, 0));
          depth_ia.push_back(counter++);

          depth_tc.push_back(glm::vec2(static_cast<float>(x    ) / static_cast<float>(internalWidth()), static_cast<float>(y    ) / static_cast<float>(internalHeight())));
          depth_tc.push_back(glm::vec2(x, y));
          depth_tc.push_back(glm::vec2(1, 1));
          depth_ia.push_back(counter++);
          depth_ia.push_back(0xffffffff); // strip-restart

        }
      }
      m_geoDepthMapPointTexcoordBuffer = rhi()->newBufferWithContents(depth_tc.data(), depth_tc.size() * sizeof(depth_tc[0]), kBufferUsageCPUWriteOnly);

      depth_ia.pop_back(); // remove the unneccesary last strip-restart index

      m_geoDepthMapPointTristripIndexBuffer = rhi()->newBufferWithContents(depth_ia.data(), depth_ia.size() * sizeof(uint32_t), kBufferUsageCPUWriteOnly);
      m_geoDepthMapPointTristripIndexCount = depth_ia.size();
    }
  }

  {
    RHIRenderPipelineDescriptor rpd;
    rpd.primitiveTopology = kPrimitiveTopologyTriangleStrip;
    rpd.primitiveRestartEnabled = true;

    RHIShaderDescriptor desc("shaders/meshDisparityDepthMap.vtx.glsl", "shaders/meshDisparityDepthMap.frag.glsl", RHIVertexLayout({
        RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "textureCoordinates", 0, sizeof(float) * 4)
      }));
    desc.addSourceFile(RHIShaderDescriptor::kGeometryShader, "shaders/meshDisparityDepthMap.geom.glsl");

    desc.setFlag("SAMPLER_TYPE", cs->cameraProvider()->rgbTextureGLSamplerType());

    m_disparityDepthMapPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), rpd);
  }

  {
    RHIRenderPipelineDescriptor rpd;
    rpd.primitiveTopology = kPrimitiveTopologyTriangleStrip;
    rpd.primitiveRestartEnabled = true;

    RHIShaderDescriptor desc("shaders/meshDisparityDepthMapPoints.vtx.glsl", "shaders/meshDisparityDepthMapPoints.frag.glsl", RHIVertexLayout({
        RHIVertexLayoutElement(0, kVertexElementTypeFloat2, "textureCoordinates",          0, 24),
        RHIVertexLayoutElement(0, kVertexElementTypeFloat2, "disparitySampleCoordinates",  8, 24),
        RHIVertexLayoutElement(0, kVertexElementTypeFloat2, "quadCoordOffset",            16, 24),

      }));

    desc.setFlag("SAMPLER_TYPE", cs->cameraProvider()->rgbTextureGLSamplerType());

    m_disparityDepthMapPointsPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), rpd);
  }

  if (!disparityMipComputePipeline) {
    disparityMipComputePipeline = rhi()->compileComputePipeline(rhi()->compileShader(RHIShaderDescriptor::computeShader("shaders/disparityMip.comp.glsl")));
  }

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
}

void DepthMapGenerator::internalRenderSetup(size_t viewIdx, bool stereo, const FxRenderView& renderView0, const FxRenderView& renderView1, const glm::mat4& modelMatrix) {
  ViewData* vd = m_viewData[viewIdx];

  if (m_usePointRendering) {
    rhi()->bindRenderPipeline(m_disparityDepthMapPointsPipeline);
    rhi()->bindStreamBuffer(0, m_geoDepthMapPointTexcoordBuffer);
  } else {
    rhi()->bindRenderPipeline(m_disparityDepthMapPipeline);
    rhi()->bindStreamBuffer(0, m_geoDepthMapTexcoordBuffer);
  }

  MeshDisparityDepthMapUniformBlock ub;
  ub.modelViewProjection[0] = renderView0.viewProjectionMatrix * modelMatrix;
  ub.modelViewProjection[1] = renderView1.viewProjectionMatrix * modelMatrix;
  ub.R1inv = vd->m_R1inv;

  ub.Q3 = vd->m_Q[0][3];
  ub.Q7 = vd->m_Q[1][3];
  ub.Q11 = vd->m_Q[2][3];
  ub.CameraDistanceMeters = vd->m_CameraDistanceMeters;
  ub.mogrify = glm::vec2(m_algoDownsampleX, m_algoDownsampleY);
  ub.disparityPrescale = disparityPrescale() * debugDisparityScale();
  ub.disparityTexLevels = vd->m_disparityTexture->mipLevels() - 1;

  ub.trim_minXY = glm::vec2(m_trimLeft, m_trimTop);
  ub.trim_maxXY = glm::vec2((vd->m_disparityTexture->width() - 1) - m_trimRight, (vd->m_disparityTexture->height() - 1) - m_trimBottom);

  ub.renderStereo = (stereo ? 1 : 0);
  ub.maxValidDisparityPixels = m_maxDisparity - 1;
  ub.maxValidDisparityRaw = static_cast<uint32_t>(static_cast<float>(m_maxDisparity - 1) / m_disparityPrescale);
  ub.maxDepthDiscontinuity = m_splitDepthDiscontinuity ? m_maxDepthDiscontinuity : FLT_MAX;

  ub.texCoordStep = glm::vec2(
    1.0f / static_cast<float>(internalWidth()),
    1.0f / static_cast<float>(internalHeight()));

  ub.minDepthCutoff = m_minDepthCutoff;
  ub.pointScale = m_pointScale;

  if (m_debugUseFixedDisparity)
    ub.debugFixedDisparity = m_debugFixedDisparityValue;
  else
    ub.debugFixedDisparity = -1;

  rhi()->loadUniformBlockImmediate(ksMeshDisparityDepthMapUniformBlock, &ub, sizeof(ub));
  rhi()->loadTexture(ksDisparityTex, vd->m_disparityTexture);

  rhi()->loadTexture(ksImageTex, m_cameraSystem->cameraProvider()->rgbTexture(vd->m_leftCameraIndex), linearClampSampler);
  rhi()->loadTexture(ksDistortionMap, m_cameraSystem->viewAtIndex(viewIdx).stereoDistortionMap[0]);
}

void DepthMapGenerator::renderDisparityDepthMapStereo(size_t viewIdx, const FxRenderView& leftRenderView, const FxRenderView& rightRenderView, const glm::mat4& modelMatrix) {
  internalRenderSetup(viewIdx, /*stereo=*/ true, leftRenderView, rightRenderView, modelMatrix);

  if (m_usePointRendering)
    rhi()->drawIndexedPrimitives(m_geoDepthMapPointTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapPointTristripIndexCount, /*indexOffsetElements=*/ 0, /*instanceCount=*/ 2);
  else
    rhi()->drawIndexedPrimitives(m_geoDepthMapTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapTristripIndexCount);
}

void DepthMapGenerator::renderDisparityDepthMap(size_t viewIdx, const FxRenderView& renderView, const glm::mat4& modelMatrix) {
  internalRenderSetup(viewIdx, /*stereo=*/ false, renderView, renderView, modelMatrix);

  if (m_usePointRendering)
    rhi()->drawIndexedPrimitives(m_geoDepthMapPointTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapPointTristripIndexCount, /*indexOffsetElements=*/ 0, /*instanceCount=*/ 1);
  else
    rhi()->drawIndexedPrimitives(m_geoDepthMapTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapTristripIndexCount);
}

void DepthMapGenerator::renderIMGUI() {
  ImGui::PushID(this);

  this->internalRenderIMGUI();

  // Common render settings -- these don't affect the algorithm.
  ImGui::Checkbox("Split depth discontinuity", &m_splitDepthDiscontinuity);
  if (m_splitDepthDiscontinuity)
    ImGui::SliderFloat("Depth Discontinuity", &m_maxDepthDiscontinuity, 0.01f, 2.0f);

  ImGui::SliderInt("Trim Left",   &m_trimLeft,   0, 64);
  ImGui::SliderInt("Trim Top",    &m_trimTop,    0, 64);
  ImGui::SliderInt("Trim Right",  &m_trimRight,  0, 64);
  ImGui::SliderInt("Trim Bottom", &m_trimBottom, 0, 64);

  ImGui::SliderFloat("Min Depth Cutoff", &m_minDepthCutoff, 0.01f, 0.30f);

  ImGui::Checkbox("Point rendering", &m_usePointRendering);
  if (m_usePointRendering)
    ImGui::SliderFloat("Point Scale", &m_pointScale, 0.5f, 3.0f);

  ImGui::Checkbox("Debug: Fixed disparity", &m_debugUseFixedDisparity);
  if (m_debugUseFixedDisparity)
    ImGui::SliderInt("Fixed Disparity", &m_debugFixedDisparityValue, 0, 256);

  ImGui::PopID();
}


void DepthMapGenerator::renderIMGUIPerformanceGraphs() {
  ImGui::PushID(this);
  this->internalRenderIMGUIPerformanceGraphs();
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

      vd->m_R1inv = glm::inverse(glmMat4FromCVMatrix(v.stereoRectification[0]));
      vd->m_Q = glmMat4FromCVMatrix(v.stereoDisparityToDepth);

      vd->m_CameraDistanceMeters = glm::length(glm::vec3(v.stereoTranslation.at<double>(0), v.stereoTranslation.at<double>(1), v.stereoTranslation.at<double>(2)));
    }

    // Let the backend impl update its derived view data components
    this->internalUpdateViewData();

    m_viewDataRevision = m_cameraSystem->calibrationDataRevision();
    uint64_t endTimeNs = currentTimeNs();
    printf("DepthMapGenerator: viewData update took %.3f ms\n", deltaTimeMs(startTimeNs, endTimeNs));
  }


  this->internalProcessFrame();
}

void DepthMapGenerator::internalGenerateDisparityMips() {
  // Filter invalid disparities: generate mip-chains

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    // Each compute shader pass generates 3 mip-levels
    uint32_t passes = (vd->m_disparityTexture->mipLevels() - 1) / 3;
    for (uint32_t pass = 0; pass < passes; ++pass) {
      rhi()->beginComputePass();
      rhi()->bindComputePipeline(disparityMipComputePipeline);

      rhi()->loadImage(ksSrcImage, vd->m_disparityTexture, kImageAccessReadOnly, pass * 3);
      rhi()->loadImage(ksDstMip1, vd->m_disparityTexture, kImageAccessWriteOnly, (pass * 3) + 1);
      rhi()->loadImage(ksDstMip2, vd->m_disparityTexture, kImageAccessWriteOnly, (pass * 3) + 2);
      rhi()->loadImage(ksDstMip3, vd->m_disparityTexture, kImageAccessWriteOnly, (pass * 3) + 3);

      DisparityMipUniformBlock ub;
      ub.sourceLevel = 0; // unused in compute shader
      ub.maxValidDisparityRaw = static_cast<uint32_t>(static_cast<float>(m_maxDisparity - 1) / m_disparityPrescale);

      rhi()->loadUniformBlockImmediate(ksDisparityMipUniformBlock, &ub, sizeof(ub));
      rhi()->dispatchCompute(
        vd->m_disparityTexture->width()  / ((1 << ((pass * 3) + 1)) * 4),
        vd->m_disparityTexture->height() / ((1 << ((pass * 3) + 1)) * 4),
        1);

      rhi()->endComputePass();
    }
  }
}

void DepthMapGenerator::ViewData::updateDisparityTexture(uint32_t w, uint32_t h, RHISurfaceFormat format) {
  m_disparityTexture = rhi()->newTexture2D(w, h, RHISurfaceDescriptor::mipDescriptor(format));
  m_disparityTextureMipTargets.clear();
  for (uint32_t level = 0; level < m_disparityTexture->mipLevels(); ++level) {
    m_disparityTextureMipTargets.push_back(rhi()->compileRenderTarget({ RHIRenderTargetDescriptorElement(m_disparityTexture, level) }));
  }

  free(m_debugCPUDisparity);
  m_debugCPUDisparity = nullptr;
  m_debugCPUDisparityBytesPerPixel = 0;
}

void DepthMapGenerator::ViewData::ensureDebugCPUAccessEnabled(uint8_t disparityBytesPerPixel) {
  if (m_debugCPUDisparity == nullptr || m_debugCPUDisparityBytesPerPixel != disparityBytesPerPixel) {
    free(m_debugCPUDisparity);

    m_debugCPUDisparityBytesPerPixel = disparityBytesPerPixel;
    m_debugCPUDisparity = malloc(m_debugCPUDisparityBytesPerPixel * m_disparityTexture->width() * m_disparityTexture->height());
  }
}


float DepthMapGenerator::debugPeekDisparityTexel(size_t viewIdx, glm::ivec2 texelCoord) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);

  texelCoord = glm::clamp(texelCoord, glm::ivec2(0, 0), glm::ivec2(vd->m_disparityTexture->width() - 1, vd->m_disparityTexture->height() - 1));

  if (vd->m_debugCPUDisparity == nullptr || vd->m_debugCPUDisparityBytesPerPixel == 0) {
    return -1.0f;
  }
  size_t pIdx = (texelCoord.y * vd->m_disparityTexture->width()) + texelCoord.x;
  uint32_t v = 0;

  switch (vd->m_debugCPUDisparityBytesPerPixel) {
    case 1: v = reinterpret_cast<uint8_t*>(vd->m_debugCPUDisparity)[pIdx]; break;
    case 2: v = reinterpret_cast<uint16_t*>(vd->m_debugCPUDisparity)[pIdx]; break;
    case 4: v = reinterpret_cast<uint32_t*>(vd->m_debugCPUDisparity)[pIdx]; break;
    default:
      assert(false && "DepthMapGenerator::debugPeekDisparity: unhandled m_debugCPUDisparityBytesPerPixel");
  }
  return static_cast<float>(v) * m_disparityPrescale;
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

  float Q3 = vd->m_Q[0][3];
  float Q7 = vd->m_Q[1][3];
  float Q11 = vd->m_Q[2][3];

  float lz = Q11 * vd->m_CameraDistanceMeters / (fDisp * m_algoDownsampleX);
  float ly = ((texelCoord.y * m_algoDownsampleY) + Q7) / Q11;
  float lx = ((texelCoord.x * m_algoDownsampleX) + Q3) / Q11;
  lx *= lz;
  ly *= lz;
  glm::vec4 pP = vd->m_R1inv * glm::vec4(lx, -ly, -lz, 1.0);
  return (glm::vec3(pP) / pP.w);
}

float DepthMapGenerator::debugComputeDepthForDisparity(size_t viewIdx, float disparityPixels) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);
  float Q11 = vd->m_Q[2][3];
  float lz = Q11 * vd->m_CameraDistanceMeters / (disparityPixels * m_algoDownsampleX);
  return lz;
}

