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
#include "common/disparityFill.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/RHICVInterop.h"
#include "rhi/gl/GLCommon.h"
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

DepthMapGenerator::DepthMapGenerator(DepthMapGeneratorBackend backend_) : m_backend(backend_) {
  memset(&m_nppStreamContext, 0, sizeof(m_nppStreamContext));
  NPP_CHECK(nppSetStream((CUstream) m_globalStream.cudaPtr()));
  NPP_CHECK(nppGetStreamContext(&m_nppStreamContext));
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
      std::vector<uint16_t> depth_tc;
      std::vector<uint32_t> depth_ia;
      depth_tc.reserve(internalWidth() * internalHeight() * 4 * 4);
      depth_ia.reserve(internalWidth() * internalHeight() * 5);
      size_t counter = 0;

      for (uint32_t y = 0; y < internalHeight(); y++) {
        for (uint32_t x = 0; x < internalWidth(); x++) {
          // [0] is disparity sample coordinates (integer texels)
          // [1] is offset in current prim (0...1 across the quad)
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
    desc.setFlag("DISPARITY_USE_FP16", m_useFP16Disparity);

    m_disparityDepthMapPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), rpd);
  }

  {
    RHIRenderPipelineDescriptor rpd;
    rpd.primitiveTopology = kPrimitiveTopologyTriangleStrip;
    rpd.primitiveRestartEnabled = true;

    RHIShaderDescriptor desc("shaders/meshDisparityDepthMapPoints.vtx.glsl", "shaders/meshDisparityDepthMapPoints.frag.glsl", RHIVertexLayout({
        RHIVertexLayoutElement(0, kVertexElementTypeUShort2, "disparitySampleCoordinates", 0, 8),
        RHIVertexLayoutElement(0, kVertexElementTypeUShort2, "quadCoordOffset",            4, 8)
      }));

    desc.setFlag("SAMPLER_TYPE", cs->cameraProvider()->rgbTextureGLSamplerType());
    desc.setFlag("DISPARITY_USE_FP16", m_useFP16Disparity);

    m_disparityDepthMapPointsPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(desc), rpd);
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

void DepthMapGenerator::internalRenderSetup(size_t viewIdx, bool stereo, const FxRenderView& renderView0, const FxRenderView& renderView1) {
  ViewData* vd = m_viewData[viewIdx];

  if (m_usePointRendering) {
    rhi()->bindRenderPipeline(m_disparityDepthMapPointsPipeline);
    rhi()->bindStreamBuffer(0, m_geoDepthMapPointTexcoordBuffer);
  } else {
    rhi()->bindRenderPipeline(m_disparityDepthMapPipeline);
    rhi()->bindStreamBuffer(0, m_geoDepthMapTexcoordBuffer);
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
  if (m_debugUseFixedDisparity)
    ub.debugFixedDisparity = m_debugFixedDisparityValue;
  else
    ub.debugFixedDisparity = -1;

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

  ub.inputImageSize = glm::vec2(m_cameraSystem->cameraProvider()->streamWidth(), m_cameraSystem->cameraProvider()->streamHeight());

  rhi()->loadUniformBlockImmediate(ksMeshDisparityDepthMapUniformBlock, &ub, sizeof(ub));
  rhi()->loadTexture(ksDisparityTex, vd->m_disparityTexture);

  rhi()->loadTexture(ksImageTex, m_cameraSystem->cameraProvider()->rgbTexture(vd->m_leftCameraIndex), linearClampSampler);
  rhi()->loadTexture(ksDistortionMap, m_cameraSystem->viewAtIndex(viewIdx).stereoDistortionMap[0]);
}

void DepthMapGenerator::renderDisparityDepthMapStereo(size_t viewIdx, const FxRenderView& leftRenderView, const FxRenderView& rightRenderView) {
  internalRenderSetup(viewIdx, /*stereo=*/ true, leftRenderView, rightRenderView);

  if (m_usePointRendering)
    rhi()->drawIndexedPrimitives(m_geoDepthMapPointTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapPointTristripIndexCount, /*indexOffsetElements=*/ 0, /*instanceCount=*/ 2);
  else
    rhi()->drawIndexedPrimitives(m_geoDepthMapTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapTristripIndexCount);
}

void DepthMapGenerator::renderDisparityDepthMap(size_t viewIdx, const FxRenderView& renderView) {
  internalRenderSetup(viewIdx, /*stereo=*/ false, renderView, renderView);

  if (m_usePointRendering)
    rhi()->drawIndexedPrimitives(m_geoDepthMapPointTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapPointTristripIndexCount, /*indexOffsetElements=*/ 0, /*instanceCount=*/ 1);
  else
    rhi()->drawIndexedPrimitives(m_geoDepthMapTristripIndexBuffer, kIndexBufferTypeUInt32, m_geoDepthMapTristripIndexCount);
}

void DepthMapGenerator::renderIMGUI() {
  ImGui::PushID(this);

  this->internalRenderIMGUI();

  // Common processing settings
  ImGui::Checkbox("Median filter", &m_useMedianFilter);
  ImGui::Checkbox("Hole-filling pass", &m_useHoleFillingPass);

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

      vd->m_R1 = glmMat4FromCVMatrix(v.stereoRectification[0]);

      vd->m_depthParameters = v.depthParameters();

      printf("DepthMapGenerator: View %zu stereoRectify depth parameters: [%.3f, %.3f, %.3f, %.3f]\n", viewIdx,
        vd->m_depthParameters[0], vd->m_depthParameters[1], vd->m_depthParameters[2], vd->m_depthParameters[3]);
    }

    // Let the backend impl update its derived view data components
    this->internalUpdateViewData();

    m_viewDataRevision = m_cameraSystem->calibrationDataRevision();
    uint64_t endTimeNs = currentTimeNs();
    printf("DepthMapGenerator: viewData update took %.3f ms\n", deltaTimeMs(startTimeNs, endTimeNs));
  }


  this->internalProcessFrame();
}

void DepthMapGenerator::internalFinalizeDisparityTexture() {

  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    if (m_useMedianFilter) {
      NppiSize dispSize;
      dispSize.width = vd->m_disparityGpuMat.cols;
      dispSize.height = vd->m_disparityGpuMat.rows;

      // Copy and expand border for median filter setup
      {
        NppiSize expandedSize;
        expandedSize.width = dispSize.width + 2;
        expandedSize.height = dispSize.height + 2;

        NPP_CHECK(nppiCopyReplicateBorder_16u_C1R_Ctx(
          /*pSrc=*/ (const Npp16u*) vd->m_disparityGpuMat.data, /*nSrcStep=*/ vd->m_disparityGpuMat.step, /*oSrcSizeROI=*/ dispSize,
          /*pDst=*/ (Npp16u*) vd->m_disparityMedianFilterSourceGpuMat.data, /*nDstStep=*/ vd->m_disparityMedianFilterSourceGpuMat.step, /*oDstSizeROI=*/ expandedSize,
          /*nTopBorderHeight=*/ 1, /*nLeftBorderWidth=*/ 1,
          m_nppStreamContext));
      }

      // Run 3x3 median filter to smooth speckles and edge discontinuities
      NppiSize oMaskSize;
      oMaskSize.width = oMaskSize.height = 3;

      NppiPoint zeroOffset;
      zeroOffset.x = zeroOffset.y = 0;

      // Offset source pointer to skip the border
      const Npp16u* pSrc = reinterpret_cast<const Npp16u*>(reinterpret_cast<uint8_t*>(vd->m_disparityMedianFilterSourceGpuMat.data) + /*offset 1 line*/ (1 * vd->m_disparityMedianFilterSourceGpuMat.step)) + /*offset 1 pixel*/ 1;

      NPP_CHECK(nppiFilterMedian_16u_C1R_Ctx(
        /*pSrc=*/ pSrc,/*nSrcStep=*/ (Npp32s) vd->m_disparityMedianFilterSourceGpuMat.step,
        /*pDst=*/ (Npp16u*) vd->m_disparityGpuMat.data, /*nDstStep=*/ (Npp32s) vd->m_disparityGpuMat.step,
        /*oSizeROI=*/ dispSize,
        /*oMaskSize=*/ oMaskSize, /*oAnchor=*/ zeroOffset,
        /*pBuffer=*/ (Npp8u*) vd->m_medianFilterScratchBuffer,
        m_nppStreamContext));
    }

    if (m_useHoleFillingPass) {
      // Filter and attempt to reconstruct invalid disparities. This writes in-place.
      float maxValidDisparityRaw = static_cast<float>(m_maxDisparity - 1) / m_disparityPrescale;
      auto chromaTex = m_cameraSystem->cameraProvider()->cudaChromaTexObject(vd->m_leftCameraIndex);

      disparityFill(chromaTex, vd->m_disparityGpuMat, maxValidDisparityRaw, vd->m_disparityMinMaxMips, (CUstream) m_globalStream.cudaPtr());
    }

    // Copy filtered disparity to render texture
    RHICUDA::copyGpuMatToSurface(vd->m_disparityGpuMat, vd->m_disparityTexture, m_globalStream);

    if (debugDisparityCPUAccessEnabled()) {
      // Copy filtered disparity to CPU-visible view
      vd->m_disparityGpuMat.download(vd->m_debugCPUDisparity, m_globalStream);
    }
  }
}


uint32_t divUp(uint32_t x, uint32_t y) {
  return (x + (y - 1)) / y;
}

void DepthMapGenerator::ViewData::updateDisparityTexture(uint32_t w, uint32_t h, RHISurfaceFormat format) {
  int cvType = 0;

  switch (format) {
    case kSurfaceFormat_R8i:
      cvType = CV_8U;
      break;

    case kSurfaceFormat_R16f:
      // CV doesn't have a dedicated type enum for fp16, so we just use uint16_t
    case kSurfaceFormat_R16i:
      cvType = CV_16U;
      break;
    default:
      assert(false && "updateDisparityTexture: unhandled RHISurfaceFormat");
  };

  m_disparityGpuMat.create(/*rows=*/ h, /*cols=*/ w, /*type=*/ cvType);

  // Pre-allocate CPU debug view, identical in size/format to GPU copy
  m_debugCPUDisparity.create(/*rows=*/ h, /*cols=*/ w, /*type=*/ cvType);

  // Pre-allocate CPU debug view of L/R inputs
  for (size_t i = 0; i < 2; ++i) {
    m_debugCPUDisparityInput[i].create(h, w, CV_8U);
  }


  // Allocate buffer for median filter
  {
    // The image must be expanded by 1px in all directions for the 3x3 median filter to succeed.
    // We use one of the nppiCopyReplicateBorder functions into this temporary
    m_disparityMedianFilterSourceGpuMat.create(/*rows=*/ h + 2, /*cols=*/ w + 2, /*type=*/ cvType);

    // The processing ROI is the original output size
    NppiSize oSizeROI;
    oSizeROI.width = w;
    oSizeROI.height = h;

    NppiSize oMaskSize;
    oMaskSize.width = oMaskSize.height = 3;

    Npp32u nBufferSize = 0;
    NPP_CHECK(nppiFilterMedianGetBufferSize_16u_C1R(oSizeROI, oMaskSize, &nBufferSize));
    printf("updateDisparityTexture(): Npp buffer for %ux%u median filter on %ux%u disparity surface is %u bytes\n",
      oMaskSize.width, oMaskSize.height, oSizeROI.width, oSizeROI.height, nBufferSize);

    CUDA_SAFE_FREE(m_medianFilterScratchBuffer);

    if (nBufferSize > 0) {
      CUDA_CHECK(cuMemAlloc(&m_medianFilterScratchBuffer, nBufferSize));
    }
  }

  // Create minmax mipchain for filtering
  // TODO parameterize pass-count
  const int passes = 2;
  m_disparityMinMaxMips.resize(passes * 3);
  printf("Disparity base %ux%u\n", w, h);

  for (size_t pass = 0; pass < m_disparityMinMaxMips.size(); ++pass) {
    uint32_t passDivider = 1 << (pass + 1);
    m_disparityMinMaxMips[pass].create(/*rows=*/ divUp(h, passDivider), /*cols=*/ divUp(w, passDivider), CV_16UC4);
    printf("Disparity MinMaxMip pass %zu: %ux%u\n", pass, m_disparityMinMaxMips[pass].cols, m_disparityMinMaxMips[pass].rows);
  }

  m_disparityTexture = rhi()->newTexture2D(w, h, RHISurfaceDescriptor(format));
}

float DepthMapGenerator::debugPeekDisparityTexel(size_t viewIdx, glm::ivec2 texelCoord) const {
  const ViewData* vd = viewDataAtIndex(viewIdx);

  if (vd->m_debugCPUDisparity.empty()) {
    return -1.0f;
  }

  texelCoord = glm::clamp(texelCoord, glm::ivec2(0, 0), glm::ivec2(vd->m_debugCPUDisparity.cols - 1, vd->m_debugCPUDisparity.rows - 1));
  float disparityRaw = 0;

  // .at(row, col) -- Y rows, X columns.
  if (m_useFP16Disparity) {
    disparityRaw = glm::unpackHalf1x16(vd->m_debugCPUDisparity.at<uint16_t>(texelCoord.y, texelCoord.x));
  } else {
    switch (vd->m_debugCPUDisparity.type()) {
      case CV_8U:  disparityRaw = static_cast<float>(vd->m_debugCPUDisparity.at<uint8_t >(texelCoord.y, texelCoord.x)); break;
      case CV_16U: disparityRaw = static_cast<float>(vd->m_debugCPUDisparity.at<uint16_t>(texelCoord.y, texelCoord.x)); break;
      default:
        assert(false && "DepthMapGenerator::debugPeekDisparity: unhandled m_debugCPUDisparity.type()");
    }
  }
  return disparityRaw * m_disparityPrescale;
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

