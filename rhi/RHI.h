#pragma once
#include "rhi/RHIBlendState.h"
#include "rhi/RHIBuffer.h"
#include "rhi/RHIComputePipeline.h"
#include "rhi/RHIDepthStencilState.h"
#include "rhi/RHIQuery.h"
#include "rhi/RHIRenderPipeline.h"
#include "rhi/RHIRenderTarget.h"
#include "rhi/RHIShader.h"
#include "rhi/RHISurface.h"
#include <stddef.h>
#include <stdint.h>
#include <map>
#include <vector>

class RHI;
namespace vr { struct Texture_t; }

RHI* rhi(); // render thread owned

void initRHI(RHI*);
void initRHIGL(); // convenience method, defined in RHI/gl/RHIGL.cpp

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

size_t RHIIndexBufferTypeSize(RHIIndexBufferType);

struct RHIRect {
  RHIRect() : x(0), y(0), width(0), height(0) {}
  static RHIRect xywh(uint32_t x, uint32_t y, uint32_t w, uint32_t h) { return RHIRect(x, y, w, h); }
  static RHIRect ltrb(uint32_t l, uint32_t t, uint32_t r, uint32_t b) { return RHIRect(l, t, r - l, b - t); }
  static RHIRect sized(uint32_t w, uint32_t h) { return RHIRect(0, 0, w, h); }

  bool empty() const { return (width == 0) && (height == 0); }

  uint32_t x, y;
  uint32_t width, height;

  uint32_t left() const { return x; }
  uint32_t top() const { return y; }
  uint32_t right() const { return x + width; }
  uint32_t bottom() const { return y + height; }
protected:
  RHIRect(uint32_t x_, uint32_t y_, uint32_t w_, uint32_t h_) : x(x_), y(y_), width(w_), height(h_) {}
};

class RHI {
public:
  virtual ~RHI();

  RHIShader::ptr compileShader(const RHIShaderDescriptor&);
  RHIRenderPipeline::ptr compileRenderPipeline(RHIShader::ptr, const RHIRenderPipelineDescriptor&);
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
  // virtual void setViewports(const RHIRect*, size_t count) = 0; // Not available on ES3
  // void setViewports(const std::vector<RHIRect>& rects) { setViewports(rects.data(), rects.size()); } // Not available on ES3
  virtual void setDepthBias(float slopeScale, float constantBias) = 0;
  virtual void bindStreamBuffer(size_t streamIndex, RHIBuffer::ptr) = 0;
  virtual void bindRenderPipeline(RHIRenderPipeline::ptr) = 0;
  virtual void bindDepthStencilState(RHIDepthStencilState::ptr) = 0;
  virtual void bindBlendState(RHIBlendState::ptr) = 0;
  virtual void setCullState(RHICullState) = 0;

  // these functions are valid between beginComputePass and endComputePass
  virtual void bindComputePipeline(RHIComputePipeline::ptr) = 0;

  // resource binding functions are valid inside a render or compute pass
  virtual void loadUniformBlock(FxAtomicString name, RHIBuffer::ptr) = 0;
  virtual void loadUniformBlockImmediate(FxAtomicString name, const void*, size_t) = 0;
  virtual void loadShaderBuffer(FxAtomicString name, RHIBuffer::ptr) = 0;

  virtual void loadTexture(FxAtomicString name, RHISurface::ptr, RHISampler::ptr sampler = RHISampler::ptr()) = 0;

  // draw execution functions for render pass
  virtual void drawPrimitives(uint32_t vertexStart, uint32_t vertexCount, uint32_t instanceCount = 1, uint32_t baseInstance = 0) = 0;
  virtual void drawPrimitivesIndirect(RHIBuffer::ptr indirectBuffer, uint32_t indirectCommandCount = 1, uint32_t indirectCommandArrayOffset = 0) = 0;
  virtual void drawIndexedPrimitives(RHIBuffer::ptr indexBuffer, RHIIndexBufferType indexBufferType, uint32_t indexCount, uint32_t indexOffsetElements = 0, uint32_t instanceCount = 1, uint32_t baseInstance = 0) = 0;
  void drawFullscreenPass();
  void drawNDCQuad();

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

  // compute dispatch functions for compute pass
  virtual void dispatchCompute(uint32_t threadgroupCountX, uint32_t threadgroupCountY = 1, uint32_t threadgroupCountZ = 1) = 0;
  virtual void dispatchComputeIndirect(RHIBuffer::ptr) = 0;

  virtual void pushDebugGroup(const char*) = 0;
  virtual void popDebugGroup() = 0;

  // misc features / caps
  virtual void swapBuffers(RHIRenderTarget::ptr) = 0;
  virtual void flush() = 0;
  virtual void flushAndWaitForGPUScheduling() = 0;
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


