#pragma once
#include "FxAtomicString.h"
#include "rhi/RHIBlendState.h"
#include "rhi/RHIBuffer.h"
#include "rhi/RHIDepthStencilState.h"
#include "rhi/RHIRenderPipeline.h"
#include "rhi/RHIShader.h"
#include "rhi/RHISurface.h"
#include <glm/glm.hpp>

// Static RHI resource objects

extern RHIRenderPipeline::ptr uiLayerPipeline;
extern RHIRenderPipeline::ptr overlayCompositePipeline;
extern RHIBuffer::ptr fullscreenPassVBO;
extern RHIBuffer::ptr ndcQuadVBO;
extern RHIBuffer::ptr frustumVisualizeVBO;
extern RHIVertexLayout fullscreenPassVertexLayout;
extern RHIVertexLayout ndcQuadVertexLayout;
extern RHIRenderPipelineDescriptor fullscreenPassPipelineDescriptor;
extern RHIRenderPipelineDescriptor ndcQuadPipelineDescriptor;

extern RHIDepthStencilState::ptr disabledDepthStencilState;
extern RHIDepthStencilState::ptr opaqueLightingPassDepthStencilState;
extern RHIDepthStencilState::ptr transparencyLightingPassDepthStencilState;
extern RHIDepthStencilState::ptr alwaysWriteDepthStencilState;
extern RHIDepthStencilState::ptr standardLessDepthStencilState;
extern RHIDepthStencilState::ptr decalDepthStencilState;
extern RHIDepthStencilState::ptr standardGreaterDepthStencilState;
extern RHIDepthStencilState::ptr readOnlyEqualDepthStencilState;
extern RHIDepthStencilState::ptr sssStencilSetupPassDepthStencilState;
extern RHIDepthStencilState::ptr sssStencilDepthStencilState;

extern RHIBlendState::ptr disabledBlendState;
extern RHIBlendState::ptr meshAlphaTestBlendState;
extern RHIBlendState::ptr standardAlphaOverBlendState;
extern RHIBlendState::ptr additiveBlendState;
extern RHIBlendState::ptr sssCompositeBlendState;
extern RHIBlendState::ptr sssDebugVisualizeBlendState;
extern RHIBlendState::ptr decalBlendState;

extern RHISampler::ptr linearClampSampler;
extern RHISampler::ptr linearMipWrapAnisoSampler;

extern FxAtomicString ksUILayerUniformBlock;
struct UILayerUniformBlock {
  glm::mat4 modelViewProjection;
};

extern FxAtomicString ksLineGizmoUniformBlock;
struct LineGizmoUniformBlock {
  glm::mat4 modelMatrix;
  glm::vec4 color;
};

