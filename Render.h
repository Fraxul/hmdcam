#pragma once
#include <glm/glm.hpp>
#include "FxAtomicString.h"
#include "rhi/RHIRenderPipeline.h"
#include "rhi/RHIRenderTarget.h"

extern glm::mat4 eyeProjection[2];
extern glm::mat4 eyeView[2];
extern RHIRenderTarget::ptr eyeRT[2];
extern RHIRenderTarget::ptr windowRenderTarget;

bool RenderInit();
void RenderShutdown();
void renderHMDFrame();
void recomputeHMDParameters();

typedef void* EGLDisplay;
typedef void* EGLContext;

EGLDisplay renderEGLDisplay();
EGLContext renderEGLContext();

struct NDCClippedQuadUniformBlock {
  glm::mat4 modelViewProjection;
  glm::vec2 minUV;
  glm::vec2 maxUV;
};
extern FxAtomicString ksNDCClippedQuadUniformBlock;

struct NDCQuadUniformBlock {
  glm::mat4 modelViewProjection;
};
extern FxAtomicString ksNDCQuadUniformBlock;

struct SolidQuadUniformBlock {
  glm::mat4 modelViewProjection;
  glm::vec4 color;
};
extern FxAtomicString ksSolidQuadUniformBlock;

extern FxAtomicString ksImageTex;
extern FxAtomicString ksLeftCameraTex;
extern FxAtomicString ksRightCameraTex;
extern FxAtomicString ksLeftDistortionMap;
extern FxAtomicString ksRightDistortionMap;
extern FxAtomicString ksOverlayTex;
extern FxAtomicString ksLeftOverlayTex;
extern FxAtomicString ksRightOverlayTex;
extern FxAtomicString ksDistortionMap;
extern FxAtomicString ksMaskTex;

extern RHIRenderPipeline::ptr camTexturedQuadPipeline;
extern RHIRenderPipeline::ptr camOverlayPipeline;
extern RHIRenderPipeline::ptr camOverlayStereoPipeline;
extern RHIRenderPipeline::ptr camOverlayStereoUndistortPipeline;
extern RHIRenderPipeline::ptr camUndistortMaskPipeline;
extern RHIRenderPipeline::ptr camGreyscalePipeline;
extern RHIRenderPipeline::ptr camGreyscaleUndistortPipeline;
extern RHIRenderPipeline::ptr solidQuadPipeline;

extern RHISurface::ptr disabledMaskTex;

