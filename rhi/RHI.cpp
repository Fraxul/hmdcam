#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include <fstream>
#include <glm/gtx/transform.hpp>

static /*CVar*/ bool rhi_dumpShaderSource = false;

static RHI* s_rhi = NULL;

void initRHIResources();

void initRHI(RHI* renderThreadRHI) {
  assert(s_rhi == NULL);

  s_rhi = renderThreadRHI;

  // Create shared render resources
  initRHIResources();
}

RHI* rhi() {
  return s_rhi;
}

// RHI static options
bool RHI::s_ndcZNearIsNegativeOne = false;
bool RHI::s_allowsLayerSelectionFromVertexShader = false;
bool RHI::s_allowsAsyncUploads = false;

RHI::~RHI() {

}

void RHI::drawFullscreenPass() {
  bindStreamBuffer(0, fullscreenPassVBO);
  drawPrimitives(0, 3);
}

void RHI::drawNDCQuad() {
  bindStreamBuffer(0, ndcQuadVBO);
  drawPrimitives(0, 4);
}

RHIShader::ptr RHI::compileShader(const RHIShaderDescriptor& descriptor) {
  uint64_t descriptorHash = descriptor.hash();
  RHIShader::ptr& cacheSlot = m_shaderCache[descriptorHash];
  if (!cacheSlot) {
    if (rhi_dumpShaderSource) {
      const std::map<RHIShaderDescriptor::ShadingUnit, std::string>& sources = descriptor.preprocessSource();
      for (std::map<RHIShaderDescriptor::ShadingUnit, std::string>::const_iterator it = sources.begin(); it != sources.end(); ++it) {
        const char* unitname = NULL;
        switch (it->first) {
          case RHIShaderDescriptor::kVertexShader: unitname = "vert"; break;
          case RHIShaderDescriptor::kGeometryShader: unitname = "geom"; break;
          case RHIShaderDescriptor::kFragmentShader: unitname = "frag"; break;
          case RHIShaderDescriptor::kTessEvaluationShader: unitname = "tese"; break;
          case RHIShaderDescriptor::kTessControlShader: unitname = "tesc"; break;
          case RHIShaderDescriptor::kComputeShader: unitname = "compute"; break;
          default: continue;
        };
        char namebuf[64];
        sprintf(namebuf, "%0.16lx.%s", descriptorHash, unitname);
        std::string outFilename = namebuf; // FxVFS::expandCachePath(std::string(namebuf)); // Removed VFS dependency

        std::ofstream of(outFilename);
        of << it->second;
        printf("RHI: wrote shader dump to %s\n", outFilename.c_str());
      }
    }
    cacheSlot = this->internalCompileShader(descriptor);
  } 
  return cacheSlot;
}

RHIRenderPipeline::ptr RHI::compileRenderPipeline(RHIShader::ptr shader, const RHIRenderPipelineDescriptor& descriptor) {
  uint64_t descriptorHash = descriptor.hash();
  RHIRenderPipeline::ptr& cacheSlot = shader->m_renderPipelineCache[descriptorHash];
  if (!cacheSlot) {
    cacheSlot = this->internalCompileRenderPipeline(shader, descriptor);
  }
  return cacheSlot;
}

RHIComputePipeline::ptr RHI::compileComputePipeline(RHIShader::ptr shader) {
  RHIComputePipeline::ptr& cacheSlot = shader->m_computePipelineCache;
  if (!cacheSlot) {
    cacheSlot = this->internalCompileComputePipeline(shader);
  }
  return cacheSlot;
}

static FxAtomicString ksSourceTex("sSourceTex");
static FxAtomicString ksBlitEmulationUniformBlock("BlitEmulationUniformBlock");
struct BlitEmulationUniformBlock {
  glm::vec2 sourceOrigin;
  glm::vec2 sourceSize;

  glm::vec2 sourceRTDimensions;
  glm::vec2 destinationOrigin;

  glm::vec2 destinationSize;
  glm::vec2 destinationRTDimensions;

  int sourceLayer;
  int pad2, pad3, pad4;;
};

void RHI::blitTex_emulated(RHIRenderTarget::ptr destinationRT, RHISurface::ptr sourceSurface, uint8_t sourceLayer, RHIRect destRect, RHIRect sourceRect) {
  if (sourceRect.empty())
    sourceRect = RHIRect::sized(sourceSurface->width(), sourceSurface->height());

  if (destRect.empty())
    destRect = RHIRect::sized(destinationRT->width(), destinationRT->height());

  bool isDepthTexture = rhiSurfaceFormatHasDepth(sourceSurface->format());

  RHIShaderDescriptor descriptor;
  descriptor.setFlag("SOURCE_SAMPLE_COUNT", static_cast<int>(sourceSurface->samples()));
  descriptor.setFlag("SOURCE_IS_ARRAY", sourceSurface->isArray());
  descriptor.setFlag("SOURCE_IS_DEPTH_TEXTURE", isDepthTexture);
  descriptor.addSourceFile(RHIShaderDescriptor::kVertexShader, "shaders/blitEmulation.vtx.glsl");
  descriptor.addSourceFile(RHIShaderDescriptor::kFragmentShader, "shaders/blitEmulation.frag.glsl");
  descriptor.setVertexLayout(ndcQuadVertexLayout);

  BlitEmulationUniformBlock uniforms;
  uniforms.sourceOrigin = glm::vec2(sourceRect.x, sourceRect.y);
  uniforms.sourceSize = glm::vec2(sourceRect.width, sourceRect.height);
  uniforms.sourceRTDimensions = glm::vec2(sourceSurface->width(), sourceSurface->height());

  uniforms.destinationOrigin = glm::vec2(destRect.x, destRect.y);
  uniforms.destinationSize = glm::vec2(destRect.width, destRect.height);
  uniforms.destinationRTDimensions = glm::vec2(destinationRT->width(), destinationRT->height());
  uniforms.sourceLayer = sourceLayer;

  if (isDepthTexture) {
    bindDepthStencilState(alwaysWriteDepthStencilState);
  } else {
    bindDepthStencilState(disabledDepthStencilState);
  }
  bindBlendState(disabledBlendState);

  RHIRenderPipelineDescriptor pipelineDesc;
  pipelineDesc.primitiveTopology = kPrimitiveTopologyTriangleStrip;
  pipelineDesc.perSampleShadingEnabled = true;

  bindRenderPipeline(compileRenderPipeline(compileShader(descriptor), pipelineDesc));
  bindStreamBuffer(0, ndcQuadVBO);
  loadTexture(ksSourceTex, sourceSurface, linearClampSampler);
  loadUniformBlockImmediate(ksBlitEmulationUniformBlock, &uniforms, sizeof(uniforms));
  drawPrimitives(0, 4, 1);
}

/*static*/ glm::mat4 RHI::adjustProjectionMatrix(const glm::mat4& m) {
  if (ndcZNearIsNegativeOne()) {
    return glm::translate(glm::vec3(0.0f, 0.0f, -1.0f)) * glm::scale(glm::vec3(1.0f, 1.0f, 2.0f)) * m;
  }
  return m;
}

void RHI::populateGlobalShaderDescriptorEnvironment(RHIShaderDescriptor* descriptor) {
  descriptor->setFlag("RHI_ALLOWS_LAYER_SELECTION_FROM_VERTEX_SHADER", s_allowsLayerSelectionFromVertexShader);
}

size_t RHIIndexBufferTypeSize(RHIIndexBufferType indexBufferType) {
  switch (indexBufferType) {
    default:
      assert(false && "RHIIndexBufferTypeSize: unhandled enum");
    case kIndexBufferTypeUInt16:
      return 2;
    case kIndexBufferTypeUInt32:
      return 4;
  };
}
