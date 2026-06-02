#pragma once
#include "rhi/RHIBlendState.h"
#include "rhi/RHIBuffer.h"
#include "rhi/RHIComputePipeline.h"
#include "rhi/RHIDepthStencilState.h"
#include "rhi/RHIFence.h"
#include "rhi/RHIQuery.h"
#include "rhi/RHIRect.h"
#include "rhi/RHIRenderPipeline.h"
#include "rhi/RHIRenderTarget.h"
#include "rhi/RHIShader.h"
#include "rhi/RHISurface.h"
#include <stddef.h>
#include <stdint.h>
#include <map>
#include <vector>

class RHI;
class RHIVulkan;
namespace vr {
struct Texture_t;
}

// Minimal set of CUDA type forward declarations
typedef struct CUstream_st* CUstream;
typedef struct CUarray_st* CUarray;
typedef unsigned long long CUtexObject_v1;
typedef CUtexObject_v1 CUtexObject;
typedef unsigned long long CUsurfObject_v1;
typedef CUsurfObject_v1 CUsurfObject;
typedef unsigned long long cudaSurfaceObject_t;
typedef unsigned long long cudaTextureObject_t;
typedef struct cudaArray* cudaArray_t;

RHI* rhi(); // render thread owned

void initRHI(RHI*);
// Install the RHI singleton without triggering initRHIResources(). Useful
// during the VK migration when some RHI methods aren't implemented yet —
// initRHIResources() would abort on the first unimplemented call.
// Phase 4+ will switch to initRHI() once buffers/states/pipelines are wired.
void installRHIInstance(RHI*);
// Tear down the RHI singleton, destroying the backend's resources. Must be
// called while the underlying GPU device is still alive (i.e. before any
// device-owning singleton like rhi()->vk() goes out of scope).
void shutdownRHI();
void initRHIGL(); // convenience method, defined in rhi/gl/RHIGL.cpp
// Top-level VK backend init, defined in rhi/vk/RHIVK.cpp.
// extraInstanceExtensions is forwarded to the underlying RHIVulkan instance
// creation (e.g. the surface-extension names returned by
// SDL_Vulkan_GetInstanceExtensions for a presenting host platform).
void initRHIVulkan(const std::vector<const char*>& extraInstanceExtensions = {});
void initRHIVulkanInteropContext(); // headless VK allocator init for the GL backend's CUDA/GL interop; defined in rhi/vk/RHIVulkan.cpp. Must run after a GL context is current and after initRHIGL().

enum RHICullState : unsigned char {
  kCullDisabled,
  kCullFrontFaces,
  kCullBackFaces
};

enum RHIRenderTargetLoadAction : unsigned char {
  kLoadUnspecified,
  kLoadPreserveContents,
  kLoadClear,
  // TargetLoadInvalidate is used when we know we're going to cover every pixel in the target
  kLoadInvalidate
};

enum RHIIndexBufferType {
  kIndexBufferTypeUInt16,
  kIndexBufferTypeUInt32
};

enum RHIImageAccessType {
  kImageAccessReadOnly,
  kImageAccessReadWrite,
  kImageAccessWriteOnly
};

size_t RHIIndexBufferTypeSize(RHIIndexBufferType);

class RHI {
public:
  virtual ~RHI();

  // Headless Vulkan allocator context, shared between presentation backends
  // and CUDA-GL interop subsystems. May return nullptr if VK initialization
  // failed; callers that require it should check at use site.
  RHIVulkan* vk() const;

  RHIShader::ptr compileShader(const RHIShaderDescriptor&);
  RHIRenderPipeline::ptr compileRenderPipeline(RHIShader::ptr, const RHIRenderPipelineDescriptor&);
  RHIRenderPipeline::ptr compileRenderPipeline(const RHIShaderDescriptor&, const RHIRenderPipelineDescriptor&);
  RHIRenderPipeline::ptr compileRenderPipeline(const char* vertexShaderFilename, const char* fragmentShaderFilename, const RHIVertexLayout& vertexLayout, RHIPrimitiveTopology);
  RHIComputePipeline::ptr compileComputePipeline(RHIShader::ptr);
  virtual RHIDepthStencilState::ptr compileDepthStencilState(const RHIDepthStencilStateDescriptor&) = 0;
  virtual RHIRenderTarget::ptr compileRenderTarget(const RHIRenderTargetDescriptor&) = 0;
  virtual RHISampler::ptr compileSampler(const RHISamplerDescriptor&) = 0;
  virtual RHIBlendState::ptr compileBlendState(const RHIBlendStateDescriptor&) = 0;

  virtual RHIBuffer::ptr newBufferWithContents(const void*, size_t, RHIBufferUsageMode = kBufferUsageCPUWriteOnly) = 0;
  virtual RHIBuffer::ptr newEmptyBuffer(size_t, RHIBufferUsageMode = kBufferUsageCPUWriteOnly) = 0;
  virtual RHIBuffer::ptr newUniformBufferWithContents(const void*, size_t) = 0;
  // virtual void clearBuffer(RHIBuffer::ptr) = 0; // Not implemented on ES3
  virtual void loadBufferData(RHIBuffer::ptr, const void*, size_t offset = 0, size_t length = 0) = 0;

  virtual RHISurface::ptr newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&) = 0;
  virtual RHISurface::ptr newTexture3D(uint32_t width, uint32_t height, uint32_t depth, const RHISurfaceDescriptor&) = 0;
  virtual RHISurface::ptr newRenderbuffer2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&) = 0;
  virtual RHISurface::ptr newHMDSwapTexture(uint32_t width, uint32_t height, const RHISurfaceDescriptor&) = 0;
  virtual void loadTextureData(RHISurface::ptr texture, RHIVertexElementType sourceDataFormat, const void* sourceData) = 0;
  virtual void generateTextureMips(RHISurface::ptr texture) = 0;
  virtual void readbackTexture(RHISurface::ptr, uint8_t layer, RHIVertexElementType dataFormat, void* outData) = 0;
  virtual void fillOpenVRTextureStruct(RHISurface::ptr, vr::Texture_t*) = 0;

  // clear value state persists between passes
  virtual void setClearColor(const glm::vec4 color) = 0;
  virtual void setClearDepth(float) = 0;
  virtual void setClearStencil(uint8_t) = 0;

  virtual void beginRenderPass(RHIRenderTarget::ptr, RHIRenderTargetLoadAction colorLoadAction, RHIRenderTargetLoadAction depthLoadAction = kLoadUnspecified, RHIRenderTargetLoadAction stencilLoadAction = kLoadUnspecified) = 0;
  virtual void endRenderPass(RHIRenderTarget::ptr) = 0;
  virtual void beginComputePass() = 0;
  virtual void endComputePass() = 0;

  // empty sourceRect uses the entire source texture dimensions, empty destRect uses the entire destination texture dimensions
  virtual void blitTex(RHISurface::ptr, uint8_t sourceLayer, RHIRect destRect = RHIRect(), RHIRect sourceRect = RHIRect()) = 0;

  // these functions are valid between beginRenderPass and endRenderPass
  virtual void setViewport(const RHIRect&) = 0;
  virtual void setViewports(const RHIRect*, size_t count) = 0;
  void setViewports(const std::vector<RHIRect>& rects) { setViewports(rects.data(), rects.size()); }
  virtual void setDepthBias(float slopeScale, float constantBias) = 0;
  virtual void bindStreamBuffer(size_t streamIndex, RHIBuffer::ptr, size_t baseOffsetBytes = 0) = 0;
  virtual void bindRenderPipeline(RHIRenderPipeline::ptr) = 0;
  virtual void bindDepthStencilState(RHIDepthStencilState::ptr) = 0;
  virtual void bindBlendState(RHIBlendState::ptr) = 0;
  virtual void setCullState(RHICullState) = 0;
  virtual void setScissorRect(const RHIRect&) = 0;
  virtual void clearScissorRect() = 0;

  // these functions are valid between beginComputePass and endComputePass
  virtual void bindComputePipeline(RHIComputePipeline::ptr) = 0;

  // resource binding functions are valid inside a render or compute pass
  virtual void loadUniformBlock(FxAtomicString name, RHIBuffer::ptr) = 0;
  virtual void loadUniformBlockImmediate(FxAtomicString name, const void*, size_t) = 0;
  virtual void loadShaderBuffer(FxAtomicString name, RHIBuffer::ptr) = 0;

  virtual void loadTexture(FxAtomicString name, RHISurface::ptr, RHISampler::ptr sampler = RHISampler::ptr()) = 0;
  // layerIndex < 0 binds the iamge as layered instead of selecting a particular layer
  virtual void loadImage(FxAtomicString name, RHISurface::ptr, RHIImageAccessType, uint32_t mipLevel = 0, int32_t layerIndex = -1) = 0;

  // draw execution functions for render pass
  virtual void drawPrimitives(uint32_t vertexStart, uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t baseInstance = 0) = 0;
  virtual void drawPrimitivesIndirect(RHIBuffer::ptr indirectBuffer, uint32_t indirectCommandCount = 1, uint32_t indirectCommandArrayOffset = 0) = 0;
  virtual void drawIndexedPrimitives(RHIBuffer::ptr indexBuffer, RHIIndexBufferType indexBufferType, uint32_t indexCount, uint32_t indexOffsetElements = 0, uint32_t instanceCount = 1, uint32_t baseInstance = 0) = 0;
  virtual void drawIndexedPrimitivesIndirect(RHIBuffer::ptr indexBuffer, RHIIndexBufferType indexBufferType, RHIBuffer::ptr indirectBuffer, uint32_t indirectCommandCount = 1, uint32_t indirectCommandArrayOffset = 0) = 0;

  // Draw a fullscreen pass -- uses a single triangle that gets clipped by the edges of the framebuffer.
  void drawFullscreenPass();
  // Draw an NDC-convention quad: XY range from -1 to 1, Z=0, UV ranges from 0 to 1.
  // (0, 0) in UV is located at (-1, -1) in XY, to make framebuffer pixel and texture coordinates line up.
  // Intended for use without a projection transform.
  void drawNDCQuad(uint32_t instanceCount = 1);
  // Draw a model-space-convention unit quad: XY range from -0.5 to 0.5, UV ranges from 0 to 1.
  // (0, 0) in UV is located at (-0.5, 0.5), to make the applied image appear upright when transformed with standard model matrix conventions.
  // Intended for use with a standard model-view-projection transform setup.
  void drawModelSpaceUnitQuad(uint32_t instanceCount = 1);

  virtual RHITimerQuery::ptr newTimerQuery() = 0;
  virtual RHIOcclusionQuery::ptr newOcclusionQuery(RHIOcclusionQueryMode queryMode) = 0;

  virtual void beginTimerQuery(RHITimerQuery::ptr) = 0;
  virtual void endTimerQuery(RHITimerQuery::ptr) = 0;
  virtual void recordTimestamp(RHITimerQuery::ptr) = 0;
  virtual void beginOcclusionQuery(RHIOcclusionQuery::ptr) = 0;
  virtual void endOcclusionQuery(RHIOcclusionQuery::ptr) = 0;

  virtual uint64_t getQueryResult(RHIOcclusionQuery::ptr) = 0;
  virtual uint64_t getQueryResult(RHITimerQuery::ptr) = 0;
  virtual uint64_t getTimestampImmediate() = 0;
  virtual bool getTimerQueryDisjointState() = 0;

  // Register a fence that becomes signaled once the GPU work recorded so far
  // this frame has completed. On the VK backend the signal is enqueued at the
  // next swapBuffers()/flush() submit -- render passes are batched into a
  // single per-frame submit, so the fence resolves at end-of-frame GPU
  // completion rather than at the end of an individual render pass. Intended
  // for cross-thread handoff of a render target to another engine (see
  // RHIFence). Returns null on backends without fence support.
  virtual RHIFence::ptr registerFrameCompletionFence();

  // compute dispatch functions for compute pass
  virtual void dispatchCompute(uint32_t threadgroupCountX, uint32_t threadgroupCountY = 1, uint32_t threadgroupCountZ = 1) = 0;
  virtual void dispatchComputeIndirect(RHIBuffer::ptr) = 0;

  virtual void pushDebugGroup(const char*) = 0;
  virtual void popDebugGroup() = 0;

  // ===== CUDA interop support =====
  // Optional feature; if supportsCUDAInterop() returns false, all other calls in this section will abort at runtime.
  virtual bool supportsCUDAInterop();

  // Call this when you're done with CUDA work and will be switching to RHI.
  // Records a CUDA-side signal on the supplied stream. The RHI consumer (RHIVK's per-frame submit) waits on the latest value.
  virtual void signalCUDAToRHI(CUstream);

  // Call this when you're done with RHI work and will be switching to CUDA.
  // Records a CUDA-side wait on the latest RHI-signaled value. The RHI side signals on flush or swapBuffers.
  virtual void signalRHIToCUDA(CUstream);

  // Allocate a buffer whose memory is shared with CUDA.
  virtual RHIBuffer::ptr newInteropBuffer(size_t sizeBytes, RHIBufferUsageMode);

  // Allocate a 2D surface whose memory is shared with CUDA.
  virtual RHISurface::ptr newInteropSurface(uint32_t width, uint32_t height, const RHISurfaceDescriptor&);


  // ===== Misc features / caps =====
  virtual void swapBuffers(RHIRenderTarget::ptr) = 0;
  virtual void flush() = 0;
  virtual void flushAndWaitForGPUScheduling() = 0;

  // Block the calling thread until all GPU work submitted through this RHI --
  // and any CUDA work feeding the CUDA<->RHI interop timeline -- has completed.
  // Intended for shutdown, before GPU-backed resources are destroyed. Default
  // is a no-op for backends with no async queue; the Vulkan backend overrides
  // it (and drains CUDA first, since VK submits can be parked on the interop
  // timeline waiting for a CUDA signal).
  virtual void waitForGPUIdle() {}

  // True once a GPU operation has reported VK_ERROR_DEVICE_LOST (set by the
  // bounded drain in waitForGPUIdle). Shutdown paths use this to skip further
  // VK teardown, which can hang indefinitely on a lost device (e.g.
  // vkDestroySwapchainKHR busy-spinning in the NVIDIA driver). Default false.
  virtual bool isDeviceLost() const { return false; }

  virtual uint32_t maxMultisampleSamples() = 0;
  virtual bool supportsGeometryShaders() = 0;
  static bool allowsLayerSelectionFromVertexShader() { return s_allowsLayerSelectionFromVertexShader; }
  static bool allowsAsyncUploads() { return s_allowsAsyncUploads; }

  static bool ndcZNearIsNegativeOne() { return s_ndcZNearIsNegativeOne; }
  static float ndcZNear() { return ndcZNearIsNegativeOne() ? -1.0f : 0.0f; }

  static glm::mat4 adjustProjectionMatrix(const glm::mat4&);

  // overridden, but children call up to parent impl
  virtual void populateGlobalShaderDescriptorEnvironment(RHIShaderDescriptor*);

protected:
  virtual RHIShader::ptr internalCompileShader(const RHIShaderDescriptor&) = 0;
  virtual RHIRenderPipeline::ptr internalCompileRenderPipeline(RHIShader::ptr, const RHIRenderPipelineDescriptor&) = 0;
  virtual RHIComputePipeline::ptr internalCompileComputePipeline(RHIShader::ptr) = 0;

  void blitTex_emulated(RHIRenderTarget::ptr, RHISurface::ptr, uint8_t sourceLayer, RHIRect destRect, RHIRect sourceRect);

  // options
  static bool s_ndcZNearIsNegativeOne;
  static bool s_allowsLayerSelectionFromVertexShader;
  static bool s_allowsAsyncUploads;

  std::map<size_t, RHIShader::ptr> m_shaderCache;
};
