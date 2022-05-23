#pragma once
#include <glm/glm.hpp>
#include "RenderBackend.h"
#include "rhi/FxAtomicString.h"
#include "rhi/RHIRect.h"
#include "rhi/RHIRenderPipeline.h"
#include "rhi/RHIRenderTarget.h"

class RenderBackend;

extern glm::mat4 eyeProjection[2];
extern glm::mat4 eyeView[2];
extern RHIRenderTarget::ptr eyeRT;
extern RHIRect eyeViewports[2];
extern RHIRenderTarget::ptr windowRenderTarget;

bool RenderInit(ERenderBackend);
void RenderShutdown();
void renderHMDFrame();
void renderSetDebugSurfaceSize(size_t x, size_t y);
RHISurface::ptr renderAcquireDebugSurface();
void renderSubmitDebugSurface(RHISurface::ptr);
const std::string& renderDebugURL();

void recomputeHMDParameters();

extern RenderBackend* renderBackend;

typedef int CUdevice;
typedef struct CUctx_st *CUcontext;
extern CUdevice cudaDevice;
extern CUcontext cudaContext;

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
extern FxAtomicString ksOverlayTex;
extern FxAtomicString ksMaskTex;
extern FxAtomicString ksDistortionMap;

extern RHIRenderPipeline::ptr camTexturedQuadPipeline;
extern RHIRenderPipeline::ptr camOverlayPipeline;
extern RHIRenderPipeline::ptr camUndistortMaskPipeline;
extern RHIRenderPipeline::ptr camUndistortOverlayPipeline;
extern RHIRenderPipeline::ptr solidQuadPipeline;

extern RHISurface::ptr disabledMaskTex;

