#pragma once
#include "rhi/RHI.h"
#include "rhi/gl/RHIBlendStateGL.h"
#include "rhi/gl/RHIBufferGL.h"
#include "rhi/gl/RHIComputePipelineGL.h"
#include "rhi/gl/RHIDepthStencilStateGL.h"
#include "rhi/gl/RHIRenderPipelineGL.h"
#include "rhi/gl/RHIRenderTargetGL.h"
#include "rhi/gl/RHIShaderGL.h"
#include <vector>

class RHIGL : public RHI {
public:
  RHIGL();
  virtual ~RHIGL();

  virtual RHIDepthStencilState::ptr compileDepthStencilState(const RHIDepthStencilStateDescriptor&);
  virtual RHIRenderTarget::ptr compileRenderTarget(const RHIRenderTargetDescriptor&);
  virtual RHISampler::ptr compileSampler(const RHISamplerDescriptor&);
  virtual RHIBlendState::ptr compileBlendState(const RHIBlendStateDescriptor&);

  virtual RHIBuffer::ptr newBufferWithContents(const void*, size_t, RHIBufferUsageMode);
  virtual RHIBuffer::ptr newEmptyBuffer(size_t, RHIBufferUsageMode);
  virtual RHIBuffer::ptr newUniformBufferWithContents(const void*, size_t);
  virtual void clearBuffer(RHIBuffer::ptr);
  virtual void loadBufferData(RHIBuffer::ptr, const void*, size_t offset, size_t length);

  virtual RHISurface::ptr newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&);
  virtual RHISurface::ptr newTexture3D(uint32_t width, uint32_t height, uint32_t depth, const RHISurfaceDescriptor&);
  virtual RHISurface::ptr newRenderbuffer2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&);
  virtual RHISurface::ptr newHMDSwapTexture(uint32_t width, uint32_t height, const RHISurfaceDescriptor&);
  virtual void loadTextureData(RHISurface::ptr texture, RHIVertexElementType sourceDataFormat, const void* sourceData);
  virtual void generateTextureMips(RHISurface::ptr texture);
  virtual void readbackTexture(RHISurface::ptr, uint8_t layer, RHIVertexElementType dataFormat, void* outData);
  virtual void fillOpenVRTextureStruct(RHISurface::ptr, vr::Texture_t*);

  virtual void setClearColor(const glm::vec4 color);
  virtual void setClearDepth(float);
  virtual void setClearStencil(uint8_t);

  virtual void beginRenderPass(RHIRenderTarget::ptr, RHIRenderTargetLoadAction colorLoadAction, RHIRenderTargetLoadAction depthLoadAction, RHIRenderTargetLoadAction stencilLoadAction);
  virtual void endRenderPass(RHIRenderTarget::ptr);
  virtual void beginComputePass();
  virtual void endComputePass();

  virtual void blitTex(RHISurface::ptr, uint8_t sourceLayer, RHIRect destRect, RHIRect sourceRect);

  virtual void setViewport(const RHIRect&);
  virtual void setViewports(const RHIRect*, size_t count);
  virtual void setDepthBias(float slopeScale, float constantBias);
  virtual void bindRenderPipeline(RHIRenderPipeline::ptr);
  virtual void bindStreamBuffer(size_t streamIndex, RHIBuffer::ptr);
  virtual void bindDepthStencilState(RHIDepthStencilState::ptr);
  virtual void bindBlendState(RHIBlendState::ptr);
  virtual void setCullState(RHICullState);

  virtual void bindComputePipeline(RHIComputePipeline::ptr);

  virtual void loadUniformBlock(FxAtomicString, RHIBuffer::ptr);
  virtual void loadUniformBlockImmediate(FxAtomicString, const void*, size_t);
  virtual void loadShaderBuffer(FxAtomicString name, RHIBuffer::ptr);
  virtual void loadTexture(FxAtomicString, RHISurface::ptr, RHISampler::ptr);

  virtual void drawPrimitives(uint32_t vertexStart, uint32_t vertexCount, uint32_t instanceCount, uint32_t baseInstance);
  virtual void drawPrimitivesIndirect(RHIBuffer::ptr indirectBuffer, uint32_t indirectCommandCount, uint32_t indirectCommandArrayOffset);
  virtual void drawIndexedPrimitives(RHIBuffer::ptr indexBuffer, RHIIndexBufferType indexBufferType, uint32_t indexCount, uint32_t indexOffsetElements, uint32_t instanceCount, uint32_t baseInstance);

  virtual RHITimerQuery::ptr newTimerQuery();
  virtual RHIOcclusionQuery::ptr newOcclusionQuery(RHIOcclusionQueryMode queryMode);

  virtual void beginTimerQuery(RHITimerQuery::ptr);
  virtual void endTimerQuery(RHITimerQuery::ptr);
  virtual void recordTimestamp(RHITimerQuery::ptr);
  virtual void beginOcclusionQuery(RHIOcclusionQuery::ptr);
  virtual void endOcclusionQuery(RHIOcclusionQuery::ptr);
  virtual uint64_t getQueryResult(RHIOcclusionQuery::ptr);
  virtual uint64_t getQueryResult(RHITimerQuery::ptr);
  virtual uint64_t getTimestampImmediate();

  virtual void dispatchCompute(uint32_t threadgroupCountX, uint32_t threadgroupCountY, uint32_t threadgroupCountZ);
  virtual void dispatchComputeIndirect(RHIBuffer::ptr);

  virtual void pushDebugGroup(const char*);
  virtual void popDebugGroup();

  // misc features / caps
  virtual void swapBuffers(RHIRenderTarget::ptr);
  virtual void flush();
  virtual void flushAndWaitForGPUScheduling();
  virtual uint32_t maxMultisampleSamples();
  virtual bool supportsGeometryShaders();

  virtual void populateGlobalShaderDescriptorEnvironment(RHIShaderDescriptor*);
protected:
  virtual RHIShader::ptr internalCompileShader(const RHIShaderDescriptor&);
  virtual RHIRenderPipeline::ptr internalCompileRenderPipeline(RHIShader::ptr, const RHIRenderPipelineDescriptor&);
  virtual RHIComputePipeline::ptr internalCompileComputePipeline(RHIShader::ptr);
  void internalPerformBlit(GLuint sourceFBO, bool sourceIsMultisampled, RHIRect destRect, RHIRect sourceRect);

  void internalSetupRenderPipelineState();

  bool inRenderPass() const { return m_activeRenderTarget.get() != NULL; }

  RHIRenderTargetGL::ptr m_activeRenderTarget;
  RHIRenderPipelineGL::ptr m_activeRenderPipeline;
  RHIComputePipelineGL::ptr m_activeComputePipeline;
  RHIBlendStateGL::ptr m_activeBlendState;
  RHIDepthStencilStateGL::ptr m_activeDepthStencilState;
  RHIBufferGL::ptr m_activeStreamBuffers[16];
  RHIBufferGL::ptr m_activeUniformBuffers[16];

  GLuint m_immediateScratchBufferId;
  char* m_immediateScratchBufferData;
  size_t m_immediateScratchBufferSize;
  size_t m_immediateScratchBufferWriteOffset;

  glm::vec4 m_clearColor;
  float m_clearDepth;
  uint8_t m_clearStencil;

  uint32_t m_uniformBufferOffsetAlignment;
  uint32_t m_maxMultisampleSamples;
  bool m_inComputePass;

  // state cache
  RHICullState m_currentCullState;
  float m_currentDepthBiasSlopeScale, m_currentDepthBiasConstant;
};

