#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#define _USE_MATH_DEFINES
#include <math.h>

static /*CVar*/ int r_textureAnisotropy = 8;

RHIRenderPipeline::ptr uiLayerPipeline;
RHIRenderPipeline::ptr overlayCompositePipeline;
RHIRenderPipeline::ptr triadGizmoPipeline;
RHIBuffer::ptr fullscreenPassVBO;
RHIBuffer::ptr ndcQuadVBO;
RHIBuffer::ptr triadGizmoVBO;
RHISurface::ptr hbaoNoiseTexture;
RHISurface::ptr emptySsaoTexture;
RHISurface::ptr emptyLayeredDepthTexture;
RHIDepthStencilState::ptr disabledDepthStencilState;
RHIDepthStencilState::ptr opaqueLightingPassDepthStencilState;
RHIDepthStencilState::ptr transparencyLightingPassDepthStencilState;
RHIDepthStencilState::ptr alwaysWriteDepthStencilState;
RHIDepthStencilState::ptr standardLessDepthStencilState;
RHIDepthStencilState::ptr decalDepthStencilState;
RHIDepthStencilState::ptr standardGreaterDepthStencilState;
RHIDepthStencilState::ptr readOnlyEqualDepthStencilState;
RHIDepthStencilState::ptr sssStencilSetupPassDepthStencilState;
RHIDepthStencilState::ptr sssStencilDepthStencilState;
RHIBlendState::ptr disabledBlendState;
RHIBlendState::ptr meshAlphaTestBlendState;
RHIBlendState::ptr standardAlphaOverBlendState;
RHIBlendState::ptr additiveBlendState;
RHIBlendState::ptr sssCompositeBlendState;
RHIBlendState::ptr sssDebugVisualizeBlendState;
RHIBlendState::ptr decalBlendState;

RHISampler::ptr linearClampSampler;
RHISampler::ptr linearMipWrapAnisoSampler;

RHIVertexLayout fullscreenPassVertexLayout;
RHIVertexLayout ndcQuadVertexLayout;
RHIRenderPipelineDescriptor tristripPipelineDescriptor;

FxAtomicString ksShadowAtlasVisualizeUniformBlock("ShadowAtlasVisualizeUniformBlock");
FxAtomicString ksHBAOUniformBlock("HBAOUniformBlock");
FxAtomicString ksBloomThresholdUniformBlock("BloomThresholdUniformBlock");
FxAtomicString ksFrustumVisualizeUniformBlock("FrustumVisualizeUniformBlock");
FxAtomicString ksLineGizmoUniformBlock("LineGizmoUniformBlock");
FxAtomicString ksUILayerUniformBlock("UILayerUniformBlock");

RHIVertexLayout positionOnlyVertexLayout;

void initRHIResources() {

  {
    // Draws a single triangle that's big enough to get clipped into a screen-filling rectangle.
    static const float fullscreenPassData[] = {
       3.0f, -1.0f, 0.0f, 2.0f, 0.0f,  // right-bottom
      -1.0f,  3.0f, 0.0f, 0.0f, 2.0f,  // left-top
      -1.0f, -1.0f, 0.0f, 0.0f, 0.0f}; // left-bottom

    fullscreenPassVBO = rhi()->newBufferWithContents(fullscreenPassData, sizeof(float) * 15);
  }

  {
    static const float ndcQuadData[] = {
       1.0f,  1.0f, 0.0f, 1.0f, 1.0f,  // right-top
       1.0f, -1.0f, 0.0f, 1.0f, 0.0f,  // right-bottom
      -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,  // left-top
      -1.0f, -1.0f, 0.0f, 0.0f, 0.0f}; // left-bottom

    ndcQuadVBO = rhi()->newBufferWithContents(ndcQuadData, sizeof(float) * 20);
  }

  ndcQuadVertexLayout.elements.clear();
  ndcQuadVertexLayout.elements.push_back(RHIVertexLayoutElement(0, kVertexElementTypeFloat3, "position",            0, sizeof(float) * 5));
  ndcQuadVertexLayout.elements.push_back(RHIVertexLayoutElement(0, kVertexElementTypeFloat2, "textureCoordinates", 12, sizeof(float) * 5));

  fullscreenPassVertexLayout.elements.clear();
  fullscreenPassVertexLayout.elements.push_back(RHIVertexLayoutElement(0, kVertexElementTypeFloat3, "position",            0, sizeof(float) * 5));
  fullscreenPassVertexLayout.elements.push_back(RHIVertexLayoutElement(0, kVertexElementTypeFloat2, "textureCoordinates", 12, sizeof(float) * 5));

  tristripPipelineDescriptor.primitiveTopology = kPrimitiveTopologyTriangleStrip;

  positionOnlyVertexLayout.elements.clear();
  positionOnlyVertexLayout.elements.push_back(RHIVertexLayoutElement(0, kVertexElementTypeFloat3, "position", 0, sizeof(glm::vec3)));

  {
    static const float triadData[] = {
      // X-axis
      0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
      1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,

      // Y-axis
      0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
      0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,

      // Z-axis

      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f,
      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f
    };

    triadGizmoVBO = rhi()->newBufferWithContents(triadData, sizeof(float) * 36);
  }

  triadGizmoPipeline = rhi()->compileRenderPipeline("shaders/triadGizmo.vtx.glsl", "shaders/triadGizmo.frag.glsl", {
      RHIVertexLayoutElement(0, kVertexElementTypeFloat3, "position",  0, 24),
      RHIVertexLayoutElement(0, kVertexElementTypeFloat3, "color", 12, 24)
    }, kPrimitiveTopologyLineList);

  // Depth-stencil states
  {
    RHIDepthStencilStateDescriptor descriptor;
    descriptor.depthTestEnable = false;
    descriptor.depthWriteEnable = false;
    descriptor.stencilTestEnable = false;
    disabledDepthStencilState = rhi()->compileDepthStencilState(descriptor);
  }

  {
    // Opaque lighting pass renders its quad at NDC Z=0 and uses the LESS test to kill lighting fragments at the far plane (skybox)
    RHIDepthStencilStateDescriptor descriptor;
    descriptor.depthTestEnable = true;
    descriptor.depthFunction = kCompareNotEqual;
    descriptor.depthWriteEnable = false;
    descriptor.stencilTestEnable = false;
    opaqueLightingPassDepthStencilState = rhi()->compileDepthStencilState(descriptor);
  }

  {
    // Transparency pass has depth initialized to one, so we render our lighting pass at Z=1 and use
    // the 'less' depth test.
    RHIDepthStencilStateDescriptor descriptor;
    descriptor.depthTestEnable = true;
    descriptor.depthFunction = kCompareGreater;
    descriptor.depthWriteEnable = false;
    descriptor.stencilTestEnable = false;
    transparencyLightingPassDepthStencilState = rhi()->compileDepthStencilState(descriptor);
  }

  {
    RHIDepthStencilStateDescriptor descriptor;
    descriptor.depthTestEnable = true;
    descriptor.depthFunction = kCompareAlways;
    descriptor.depthWriteEnable = true;
    descriptor.stencilTestEnable = false;
    alwaysWriteDepthStencilState = rhi()->compileDepthStencilState(descriptor);
  }

  {
    RHIDepthStencilStateDescriptor descriptor;
    descriptor.depthTestEnable = true;
    descriptor.depthFunction = kCompareLess;
    descriptor.depthWriteEnable = true;
    descriptor.stencilTestEnable = false;
    standardLessDepthStencilState = rhi()->compileDepthStencilState(descriptor);
  }

  {
    RHIDepthStencilStateDescriptor descriptor;
    descriptor.depthTestEnable = true;
    descriptor.depthFunction = kCompareGreaterEqual;
    descriptor.depthWriteEnable = false;
    descriptor.stencilTestEnable = false;
    decalDepthStencilState = rhi()->compileDepthStencilState(descriptor);
  }

  {
    RHIDepthStencilStateDescriptor descriptor;
    descriptor.depthTestEnable = true;
    descriptor.depthFunction = kCompareGreater;
    descriptor.depthWriteEnable = true;
    descriptor.stencilTestEnable = false;
    standardGreaterDepthStencilState = rhi()->compileDepthStencilState(descriptor);
  }

  {
    RHIDepthStencilStateDescriptor descriptor;
    descriptor.depthTestEnable = true;
    descriptor.depthFunction = kCompareEqual;
    descriptor.depthWriteEnable = false;
    descriptor.stencilTestEnable = false;
    readOnlyEqualDepthStencilState = rhi()->compileDepthStencilState(descriptor);
  }

  {
    RHIDepthStencilStateDescriptor descriptor;
    descriptor.depthTestEnable = false;
    descriptor.depthWriteEnable = false;
    descriptor.stencilTestEnable = true;

    descriptor.stencilFront.failOp = kStencilKeep;
    descriptor.stencilFront.depthFailOp = kStencilZero;
    descriptor.stencilFront.passOp = kStencilReplace;
    descriptor.stencilFront.compareFunc = kCompareAlways;
    descriptor.stencilFront.referenceValue = 1;

    descriptor.stencilBack = descriptor.stencilFront;

    sssStencilSetupPassDepthStencilState = rhi()->compileDepthStencilState(descriptor);
  }

  {
    RHIDepthStencilStateDescriptor descriptor;
    descriptor.depthTestEnable = false;
    descriptor.depthWriteEnable = false;
    descriptor.stencilTestEnable = true;

    descriptor.stencilFront.failOp = kStencilKeep;
    descriptor.stencilFront.depthFailOp = kStencilKeep;
    descriptor.stencilFront.passOp = kStencilKeep;
    descriptor.stencilFront.compareFunc = kCompareNotEqual;
    descriptor.stencilFront.referenceValue = 0;

    descriptor.stencilBack = descriptor.stencilFront;

    sssStencilDepthStencilState = rhi()->compileDepthStencilState(descriptor);
  }

  {
    disabledBlendState = rhi()->compileBlendState(RHIBlendStateDescriptor());
  }

  {
    RHIBlendStateDescriptor descriptor(RHIBlendStateDescriptorElement(kBlendOne, kBlendZero, kBlendZero, kBlendZero, kBlendFuncAdd));
    meshAlphaTestBlendState = rhi()->compileBlendState(descriptor);
  }

  {
    RHIBlendStateDescriptorElement alphaOverEl(kBlendSourceAlpha, kBlendOneMinusSourceAlpha, kBlendFuncAdd);
    RHIBlendStateDescriptor descriptor(alphaOverEl);
    standardAlphaOverBlendState = rhi()->compileBlendState(descriptor);
  }

  additiveBlendState = rhi()->compileBlendState(RHIBlendStateDescriptor(RHIBlendStateDescriptorElement(kBlendOne, kBlendOne, kBlendFuncAdd)));

  sssCompositeBlendState = rhi()->compileBlendState(RHIBlendStateDescriptor(RHIBlendStateDescriptorElement(kBlendDestAlpha, kBlendOneMinusDestAlpha, kBlendZero, kBlendOne, kBlendFuncAdd)));

  sssDebugVisualizeBlendState = rhi()->compileBlendState(RHIBlendStateDescriptor(RHIBlendStateDescriptorElement(kBlendSourceAlpha, kBlendZero, kBlendZero, kBlendOne, kBlendFuncAdd)));

  {
    // decal blending uses source alpha to blend colors in the color buffer, preserving destination alpha (subsurface coefficient).
    RHIBlendStateDescriptor descriptor(RHIBlendStateDescriptorElement(kBlendSourceAlpha, kBlendOneMinusSourceAlpha, kBlendZero, kBlendOne));
    decalBlendState = rhi()->compileBlendState(descriptor);
  }

  {
    RHISamplerDescriptor descriptor;
    descriptor.filter = kFilterLinear;
    linearClampSampler = rhi()->compileSampler(descriptor);
  }

  {
    RHISamplerDescriptor descriptor;
    descriptor.wrapModeU = kWrapRepeat;
    descriptor.wrapModeV = kWrapRepeat;
    descriptor.filter = kFilterMipLinear;
    descriptor.maxAnisotropy = r_textureAnisotropy;
    linearMipWrapAnisoSampler = rhi()->compileSampler(descriptor);
  }

  // Set up shaders and static pipelines

  RHIShaderDescriptor uiLayerShaderDescriptor;
  uiLayerShaderDescriptor.addSourceFile(RHIShaderDescriptor::kVertexShader, "shaders/uiLayer.vtx.glsl");
  uiLayerShaderDescriptor.addSourceFile(RHIShaderDescriptor::kFragmentShader, "shaders/uiLayer.frag.glsl");
  uiLayerShaderDescriptor.setVertexLayout(ndcQuadVertexLayout);
  uiLayerPipeline = rhi()->compileRenderPipeline(rhi()->compileShader(uiLayerShaderDescriptor), tristripPipelineDescriptor);

  RHIShaderDescriptor overlayCompositeShaderDescriptor;
  overlayCompositeShaderDescriptor.addSourceFile(RHIShaderDescriptor::kVertexShader, "shaders/lightPass.vtx.glsl");
  overlayCompositeShaderDescriptor.addSourceFile(RHIShaderDescriptor::kFragmentShader, "shaders/overlayCompositeShader.frag.glsl");
  overlayCompositeShaderDescriptor.setVertexLayout(ndcQuadVertexLayout);
  overlayCompositePipeline = rhi()->compileRenderPipeline(rhi()->compileShader(overlayCompositeShaderDescriptor), tristripPipelineDescriptor);
}

FxAtomicString ksTriadGizmoUniformBuffer("TriadGizmoUniformBuffer");
FxAtomicString ksPositionDataBuffer("PositionDataBuffer");

void drawTriadGizmosForPoints(RHIBuffer::ptr pointBuf, size_t count, const glm::mat4& viewProjection) {
  rhi()->bindRenderPipeline(triadGizmoPipeline);
  rhi()->bindStreamBuffer(0, triadGizmoVBO);
  rhi()->loadShaderBuffer(ksPositionDataBuffer, pointBuf);
  rhi()->loadUniformBlockImmediate(ksTriadGizmoUniformBuffer, &viewProjection, sizeof(glm::mat4));
  rhi()->drawPrimitives(0, 6, count);
}

