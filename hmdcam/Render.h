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

// Expected use of this is to compare against kUserPresenceState_NotPresent --
// if the presence state is Unknown, it's usually better to assume the user is present.
enum UserPresenceState {
  kUserPresenceState_NotPresent,
  kUserPresenceState_Present,
  kUserPresenceState_Unknown
};

UserPresenceState RenderGetUserPresenceState();
void renderHMDFrame();

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
  glm::mat4 modelViewProjection[2];
  glm::vec4 color;
};
extern FxAtomicString ksSolidQuadUniformBlock;

extern FxAtomicString ksOverlayTex;
extern FxAtomicString ksMaskTex;
extern FxAtomicString ksDistortionMap;

extern RHIRenderPipeline::ptr camTexturedQuadPipeline;
extern RHIRenderPipeline::ptr camOverlayPipeline;
extern RHIRenderPipeline::ptr camUndistortMaskPipeline;
extern RHIRenderPipeline::ptr camUndistortOverlayPipeline;
extern RHIRenderPipeline::ptr solidQuadPipeline;

extern RHISurface::ptr disabledMaskTex;

