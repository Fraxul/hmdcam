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
  } else if ((!strcasecmp(backendStr, "vpi"))) {
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

RHIRenderPipeline::ptr disparityDepthMapPipeline;
FxAtomicString ksMeshDisparityDepthMapUniformBlock("MeshDisparityDepthMapUniformBlock");
static FxAtomicString ksDisparityTex("disparityTex");
static FxAtomicString ksImageTex("imageTex");
extern FxAtomicString ksDistortionMap;
struct MeshDisparityDepthMapUniformBlock {
  glm::mat4 modelViewProjection[2];
  glm::mat4 R1inv;
  float Q3, Q7, Q11;
  float CameraDistanceMeters;

  glm::vec2 mogrify;
  float disparityPrescale;
  int disparityTexLevels;

  glm::vec2 trim_minXY;
  glm::vec2 trim_maxXY;

  int renderStereo;
  float pad2, pad3, pad4;
};

RHIRenderPipeline::ptr disparityMipPipeline;
FxAtomicString ksDisparityMipUniformBlock("DisparityMipUniformBlock");
struct DisparityMipUniformBlock {
  uint32_t sourceLevel;
  float pad2, pad3, pad4;
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

/*
Vector4 DepthMapGenerator::TransformToLocalSpace(float x, float y, int disp)
{
  float fDisp = (float) disp / 16.f; //  16-bit fixed-point disparity map (where each disparity value has 4 fractional bits)

  float lz = m_Q[11] * m_CameraDistanceMeters / (fDisp * m_algoDownsampleX);
  float ly = -(y * m_algoDownsampleY + m_Q[7]) / m_Q[11];
  float lx = (x * m_algoDownsampleX + m_Q[3]) / m_Q[11];
  lx *= lz;
  ly *= lz;
  lz *= -1;
  return m_R1inv * Vector4(lx, ly, lz, 1.0);
}
*/

void DepthMapGenerator::internalRenderSetup(size_t viewIdx, bool stereo, const FxRenderView& renderView0, const FxRenderView& renderView1, const glm::mat4& modelMatrix) {
  if (!disparityDepthMapPipeline) {
    RHIRenderPipelineDescriptor rpd;
    rpd.primitiveTopology = kPrimitiveTopologyTriangleStrip;
    rpd.primitiveRestartEnabled = true;

    RHIShaderDescriptor desc("shaders/meshDisparityDepthMap.vtx.glsl", "shaders/meshDisparityDepthMap.frag.glsl", RHIVertexLayout({
        RHIVertexLayoutElement(0, kVertexElementTypeFloat4, "textureCoordinates", 0, sizeof(float) * 4)
      }));
    desc.addSourceFile(RHIShaderDescriptor::kGeometryShader, "shaders/meshDisparityDepthMap.geom.glsl");

    if (epoxy_is_desktop_gl()) // TODO query this at use-time from the RHISurface type
      desc.setFlag("SAMPLER_TYPE", "sampler2D");
    else
      desc.setFlag("SAMPLER_TYPE", "samplerExternalOES");

    disparityDepthMapPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), rpd);
  }

  ViewData* vd = m_viewData[viewIdx];

  rhi()->bindRenderPipeline(disparityDepthMapPipeline);
  rhi()->bindStreamBuffer(0, m_geoDepthMapTexcoordBuffer);

  MeshDisparityDepthMapUniformBlock ub;
  ub.modelViewProjection[0] = renderView0.viewProjectionMatrix * modelMatrix;
  ub.modelViewProjection[1] = renderView1.viewProjectionMatrix * modelMatrix;
  ub.R1inv = vd->m_R1inv;

  ub.Q3 = vd->m_Q[0][3];
  ub.Q7 = vd->m_Q[1][3];
  ub.Q11 = vd->m_Q[2][3];
  ub.CameraDistanceMeters = vd->m_CameraDistanceMeters;
  ub.mogrify = glm::vec2(m_algoDownsampleX, m_algoDownsampleY);
  ub.disparityPrescale = m_disparityPrescale;
  ub.disparityTexLevels = vd->m_disparityTexture->mipLevels() - 1;

  ub.trim_minXY = glm::vec2(m_trimLeft, m_trimTop);
  ub.trim_maxXY = glm::vec2((vd->m_disparityTexture->width() - 1) - m_trimRight, (vd->m_disparityTexture->height() - 1) - m_trimBottom);

  ub.renderStereo = (stereo ? 1 : 0);

  rhi()->loadUniformBlockImmediate(ksMeshDisparityDepthMapUniformBlock, &ub, sizeof(ub));
  rhi()->loadTexture(ksDisparityTex, vd->m_disparityTexture);

  rhi()->loadTexture(ksImageTex, m_cameraSystem->cameraProvider()->rgbTexture(vd->m_leftCameraIndex), linearClampSampler);
  rhi()->loadTexture(ksDistortionMap, m_cameraSystem->viewAtIndex(viewIdx).stereoDistortionMap[0]);
}

void DepthMapGenerator::renderDisparityDepthMapStereo(size_t viewIdx, const FxRenderView& leftRenderView, const FxRenderView& rightRenderView, const glm::mat4& modelMatrix) {
  internalRenderSetup(viewIdx, /*stereo=*/ true, leftRenderView, rightRenderView, modelMatrix);
  rhi()->drawIndexedPrimitives(m_geoDepthMapTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapTristripIndexCount);
}

void DepthMapGenerator::renderDisparityDepthMap(size_t viewIdx, const FxRenderView& renderView, const glm::mat4& modelMatrix) {
  internalRenderSetup(viewIdx, /*stereo=*/ false, renderView, renderView, modelMatrix);
  rhi()->drawIndexedPrimitives(m_geoDepthMapTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapTristripIndexCount);
}

void DepthMapGenerator::renderIMGUI() {
  ImGui::PushID(this);

  this->internalRenderIMGUI();

  // Common render settings -- these don't affect the algorithm.
  ImGui::SliderInt("Trim Left",   &m_trimLeft,   0, 64);
  ImGui::SliderInt("Trim Top",    &m_trimTop,    0, 64);
  ImGui::SliderInt("Trim Right",  &m_trimRight,  0, 64);
  ImGui::SliderInt("Trim Bottom", &m_trimBottom, 0, 64);
  ImGui::PopID();
}

void DepthMapGenerator::processFrame() {

  // Update view data
  if (m_viewData.empty() || (m_viewDataRevision != m_cameraSystem->calibrationDataRevision())) {

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
  }


  this->internalProcessFrame();
}

void DepthMapGenerator::internalGenerateDisparityMips() {
  if (!disparityMipPipeline) {
    disparityMipPipeline = rhi()->compileRenderPipeline("shaders/ndcQuad.vtx.glsl", "shaders/disparityMip.frag.glsl", ndcQuadVertexLayout, kPrimitiveTopologyTriangleStrip);
  }

  // Filter invalid disparities: generate mip-chains
  uint32_t maxLevels = 0;
  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (vd->m_isStereoView)
      maxLevels = std::max<uint32_t>(maxLevels, vd->m_disparityTexture->mipLevels());
  }

  // Organized by mip level to give the driver a chance at overlapping the render passes
  for (uint32_t targetLevel = 1; targetLevel < maxLevels; ++targetLevel) {
    for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
      auto vd = viewDataAtIndex(viewIdx);
      if (!vd->m_isStereoView || vd->m_disparityTexture->mipLevels() < targetLevel)
        continue;

      rhi()->beginRenderPass(vd->m_disparityTextureMipTargets[targetLevel], kLoadInvalidate);
      rhi()->bindRenderPipeline(disparityMipPipeline);
      rhi()->loadTexture(ksImageTex, vd->m_disparityTexture);

      DisparityMipUniformBlock ub;
      ub.sourceLevel = targetLevel - 1;

      rhi()->loadUniformBlockImmediate(ksDisparityMipUniformBlock, &ub, sizeof(ub));
      rhi()->drawNDCQuad();
      rhi()->endRenderPass(vd->m_disparityTextureMipTargets[targetLevel]);
    }
  }
}

void DepthMapGenerator::ViewData::updateDisparityTexture(uint32_t w, uint32_t h, RHISurfaceFormat format) {
  m_disparityTexture = rhi()->newTexture2D(w, h, RHISurfaceDescriptor::mipDescriptor(format));
  m_disparityTextureMipTargets.clear();
  for (uint32_t level = 0; level < m_disparityTexture->mipLevels(); ++level) {
    m_disparityTextureMipTargets.push_back(rhi()->compileRenderTarget({ RHIRenderTargetDescriptorElement(m_disparityTexture, level) }));
  }
}

