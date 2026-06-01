#pragma once
#include "rhi/RHI.h"
#include "rhi/vk/RHIFenceVK.h"
#include "rhi/vk/RHIVulkan.h"
#include <cuda_runtime.h>
#include <condition_variable>
#include <deque>
#include <glm/glm.hpp>
#include <mutex>
#include <thread>

class VKFrameSource;
class RHIWindowRenderTargetVK;
class RHIRenderPipelineVK;

class RHIVK : public RHI {
public:
  RHIVK();
  virtual ~RHIVK();

  // Attach the swapchain-owning frame source (implemented by the render
  // backend). Must be called after the backend's createPresentation() has
  // run so the swap image count is known. Allocates the command pool +
  // per-frame ring (one slot per swap image).
  void setFrameSource(VKFrameSource* source);

  // Drop cached resources tied to the previous swapchain's VkImages. The
  // frame source must call this after a swapchain recreation (and after
  // device.waitIdle()) so that stale VkImageView handles aren't held against
  // images the driver has destroyed. The frame slots themselves (command
  // buffers + fences + transient ring) are retained — swap-image count is
  // assumed stable across the recreation.
  void invalidateSwapchainResources();

  virtual RHIDepthStencilState::ptr compileDepthStencilState(const RHIDepthStencilStateDescriptor&) override;
  virtual RHIRenderTarget::ptr compileRenderTarget(const RHIRenderTargetDescriptor&) override;
  virtual RHISampler::ptr compileSampler(const RHISamplerDescriptor&) override;
  virtual RHIBlendState::ptr compileBlendState(const RHIBlendStateDescriptor&) override;

  virtual RHIBuffer::ptr newBufferWithContents(const void*, size_t, RHIBufferUsageMode) override;
  virtual RHIBuffer::ptr newEmptyBuffer(size_t, RHIBufferUsageMode) override;
  virtual RHIBuffer::ptr newUniformBufferWithContents(const void*, size_t) override;
  virtual void loadBufferData(RHIBuffer::ptr, const void*, size_t offset, size_t length) override;

  virtual RHISurface::ptr newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&) override;
  virtual RHISurface::ptr newTexture3D(uint32_t width, uint32_t height, uint32_t depth, const RHISurfaceDescriptor&) override;
  virtual RHISurface::ptr newRenderbuffer2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor&) override;
  virtual RHISurface::ptr newHMDSwapTexture(uint32_t width, uint32_t height, const RHISurfaceDescriptor&) override;
  virtual void loadTextureData(RHISurface::ptr texture, RHIVertexElementType sourceDataFormat, const void* sourceData) override;
  virtual void generateTextureMips(RHISurface::ptr texture) override;
  virtual void readbackTexture(RHISurface::ptr, uint8_t layer, RHIVertexElementType dataFormat, void* outData) override;
  virtual void fillOpenVRTextureStruct(RHISurface::ptr, vr::Texture_t*) override;

  virtual void setClearColor(const glm::vec4 color) override;
  virtual void setClearDepth(float) override;
  virtual void setClearStencil(uint8_t) override;

  virtual void beginRenderPass(RHIRenderTarget::ptr, RHIRenderTargetLoadAction colorLoadAction, RHIRenderTargetLoadAction depthLoadAction, RHIRenderTargetLoadAction stencilLoadAction) override;
  virtual void endRenderPass(RHIRenderTarget::ptr) override;
  virtual void beginComputePass() override;
  virtual void endComputePass() override;

  virtual void blitTex(RHISurface::ptr, uint8_t sourceLayer, RHIRect destRect, RHIRect sourceRect) override;

  virtual void setViewport(const RHIRect&) override;
  virtual void setViewports(const RHIRect*, size_t count) override;
  virtual void setDepthBias(float slopeScale, float constantBias) override;
  virtual void bindRenderPipeline(RHIRenderPipeline::ptr) override;
  virtual void bindStreamBuffer(size_t streamIndex, RHIBuffer::ptr, size_t baseOffsetBytes) override;
  virtual void bindDepthStencilState(RHIDepthStencilState::ptr) override;
  virtual void bindBlendState(RHIBlendState::ptr) override;
  virtual void setCullState(RHICullState) override;
  virtual void setScissorRect(const RHIRect&) override;
  virtual void clearScissorRect() override;

  virtual void bindComputePipeline(RHIComputePipeline::ptr) override;

  virtual void loadUniformBlock(FxAtomicString name, RHIBuffer::ptr) override;
  virtual void loadUniformBlockImmediate(FxAtomicString name, const void*, size_t) override;
  virtual void loadShaderBuffer(FxAtomicString name, RHIBuffer::ptr) override;
  virtual void loadTexture(FxAtomicString name, RHISurface::ptr, RHISampler::ptr sampler) override;
  virtual void loadImage(FxAtomicString name, RHISurface::ptr, RHIImageAccessType, uint32_t mipLevel, int32_t layerIndex) override;

  virtual void drawPrimitives(uint32_t vertexStart, uint32_t vertexCount, uint32_t instanceCount, uint32_t baseInstance) override;
  virtual void drawPrimitivesIndirect(RHIBuffer::ptr indirectBuffer, uint32_t indirectCommandCount, uint32_t indirectCommandArrayOffset) override;
  virtual void drawIndexedPrimitives(RHIBuffer::ptr indexBuffer, RHIIndexBufferType indexBufferType, uint32_t indexCount, uint32_t indexOffsetElements, uint32_t instanceCount, uint32_t baseInstance) override;
  virtual void drawIndexedPrimitivesIndirect(RHIBuffer::ptr indexBuffer, RHIIndexBufferType indexBufferType, RHIBuffer::ptr indirectBuffer, uint32_t indirectCommandCount, uint32_t indirectCommandArrayOffset) override;

  virtual RHITimerQuery::ptr newTimerQuery() override;
  virtual RHIOcclusionQuery::ptr newOcclusionQuery(RHIOcclusionQueryMode queryMode) override;

  virtual void beginTimerQuery(RHITimerQuery::ptr) override;
  virtual void endTimerQuery(RHITimerQuery::ptr) override;
  virtual void recordTimestamp(RHITimerQuery::ptr) override;
  virtual void beginOcclusionQuery(RHIOcclusionQuery::ptr) override;
  virtual void endOcclusionQuery(RHIOcclusionQuery::ptr) override;
  virtual uint64_t getQueryResult(RHIOcclusionQuery::ptr) override;
  virtual uint64_t getQueryResult(RHITimerQuery::ptr) override;
  virtual uint64_t getTimestampImmediate() override;
  virtual bool getTimerQueryDisjointState() override;

  virtual RHIFence::ptr registerFrameCompletionFence() override;

  virtual void dispatchCompute(uint32_t threadgroupCountX, uint32_t threadgroupCountY, uint32_t threadgroupCountZ) override;
  virtual void dispatchComputeIndirect(RHIBuffer::ptr) override;

  virtual void pushDebugGroup(const char*) override;
  virtual void popDebugGroup() override;

  virtual void swapBuffers(RHIRenderTarget::ptr) override;
  virtual void flush() override;
  virtual void flushAndWaitForGPUScheduling() override;
  virtual uint32_t maxMultisampleSamples() override;
  virtual bool supportsGeometryShaders() override;

  // CUDA interop support

  virtual bool supportsCUDAInterop() override;
  virtual void signalCUDAToRHI(CUstream) override;
  virtual void signalRHIToCUDA(CUstream) override;
  virtual RHIBuffer::ptr newInteropBuffer(size_t, RHIBufferUsageMode) override;
  virtual RHISurface::ptr newInteropSurface(uint32_t width, uint32_t height, const RHISurfaceDescriptor&) override;

protected:
  virtual RHIShader::ptr internalCompileShader(const RHIShaderDescriptor&) override;
  virtual RHIRenderPipeline::ptr internalCompileRenderPipeline(RHIShader::ptr, const RHIRenderPipelineDescriptor&) override;
  virtual RHIComputePipeline::ptr internalCompileComputePipeline(RHIShader::ptr) override;

  // -------- Phase 3: frame / command buffer state --------

  VKFrameSource* m_frameSource = nullptr;

  // Command pool with per-buffer reset support; one command buffer + fence
  // per swap image, sized to allow that many frames in flight.
  vk::UniqueCommandPool m_commandPool;
  struct FrameContext {
    vk::UniqueCommandBuffer commandBuffer;
    vk::UniqueFence frameFence;
    // Resources referenced by this frame's CB. We hold a refcounted
    // pointer here so that callers' transient RHIBuffer::ptr can go out
    // of scope without the underlying VkBuffer being destroyed before
    // the GPU has finished consuming it. Cleared at the start of each
    // new frame on this slot (after waiting on frameFence).
    std::vector<boost::intrusive_ptr<RHIObject>> keepAlive;
  };
  std::vector<FrameContext> m_frameContexts;
  size_t m_frameContextIndex = 0;

  // Image views created on-demand for swap images. Keyed by VkImage handle
  // so we reuse views across the lifetime of the swap chain.
  std::map<VkImage, vk::UniqueImageView> m_swapImageViews;
  vk::ImageView swapImageViewFor(vk::Image image, vk::Format format);

  // Frame state. Lifecycle:
  //   - cbOpen=false at process start and immediately after swapBuffers/flush.
  //   - First begin* call (renderPass, timerQuery, computePass) calls
  //     ensureFrameCBOpen() which waits on the slot fence, recycles the
  //     slot's resources, and begins recording into the slot's CB.
  //   - Multiple render passes record into the same CB.
  //   - swapBuffers / flush() ends the CB and submits.
  //   - passActive bracket-tracks the *currently recording render pass*
  //     (between beginRenderPass and endRenderPass) for validation +
  //     for endRenderPass to know which image to barrier.
  // The window-RT swap-image state lives here once a window-RT pass has
  // been entered; swapBuffers reads it back out for submit + present.
  struct CurrentFrame {
    // ---- Frame scope (set by ensureFrameCBOpen, cleared by swap/flush) ----
    bool cbOpen = false;
    vk::CommandBuffer commandBuffer; // borrowed handle into the slot's CB
    vk::Fence frameFence; // borrowed handle into the slot's fence

    // ---- Window-RT scope (set by beginRenderPass(window), cleared by swap) ----
    bool hasWindowImage = false;
    uint32_t swapchainIndex = 0;
    vk::Image swapImage;
    vk::Extent2D swapExtent{};
    vk::Format swapFormat{};
    vk::Semaphore imageAcquired;
    vk::Semaphore renderFinished;

    // ---- Render-pass scope (set by beginRenderPass, cleared by endRenderPass) ----
    bool passActive = false;
    bool passIsWindowRT = false;
    vk::Extent2D passExtent{};

    // All color attachments bound this pass (0..N). A window-RT pass holds
    // exactly the acquired swap image; an offscreen pass mirrors the render
    // target's color attachment list (which may be empty for a depth-only
    // pass).
    struct PassColorAttachment {
      vk::Image image;
      vk::ImageView view;
      vk::Format format{};
      vk::ImageLayout targetLayout{vk::ImageLayout::eShaderReadOnlyOptimal};
    };
    std::vector<PassColorAttachment> passColorAttachments;

    // Optional depth-stencil attachment. hasDepth / hasStencil are derived
    // from the surface's RHISurfaceFormat so begin/endRenderPass can select
    // the image aspect and decide which of pDepthAttachment / pStencilAttachment
    // to populate.
    bool passHasDepthStencil = false;
    bool passDepthStencilHasDepth = false;
    bool passDepthStencilHasStencil = false;
    vk::Image passDepthStencilImage;
    vk::ImageView passDepthStencilView;

    // Primary color attachment — aliases passColorAttachments[0] when present,
    // null for a depth-only pass. Retained for the debug PNG-dump path and
    // window-RT bookkeeping (which always has exactly one color attachment).
    vk::Image passImage;
    vk::ImageView passImageView;
    vk::Format passFormat{};
    vk::ImageLayout passImageLayout{};

    // Number of viewports currently set. shader_object requires the scissor
    // count to match the viewport count at draw time, so the scissor setters
    // replicate across this count. Reset to 1 by setDynamicStateDefaults.
    uint32_t viewportCount = 1;

    // ---- Frame scope: pending PNG dumps (debug, env-var-gated) ----
    // One entry per endRenderPass on a dump frame. Drained in swapBuffers /
    // flush after the queue submit has completed (waitIdle), then the
    // CurrentFrame reset destroys the staging buffers.
    struct PendingDump {
      vk::UniqueBuffer buffer;
      vk::UniqueDeviceMemory memory;
      vk::DeviceSize size = 0;
      uint32_t width = 0;
      uint32_t height = 0;
      vk::Format format{};
      std::string filepath;
    };
    std::vector<PendingDump> pendingDumps;
  };
  CurrentFrame m_currentFrame;

  // Open the frame's CB if not already open: wait on the slot fence, reset
  // the fence + transient ring + keepAlive, begin the slot's command buffer.
  // Idempotent within a frame — called from every begin* path.
  void ensureFrameCBOpen();

  glm::vec4 m_clearColor{0.0f, 0.0f, 0.0f, 1.0f};
  float m_clearDepth = 1.0f;
  uint8_t m_clearStencil = 0;

  // -------- Phase 4: bind-state tracking --------
  //
  // shader_object expects every piece of dynamic state to be set per draw —
  // pipelines no longer bake any of it in. bindRenderPipeline establishes
  // safe defaults, and the bindXState/setX methods overwrite individual
  // pieces. draw* assumes everything is set.
  //
  // Resource binding (loadTexture/loadUniformBlock/...) is staged into
  // m_pendingPushDescriptors and flushed via vkCmdPushDescriptorSetKHR at
  // draw time. Stream-buffer binds are similarly staged and flushed.

  RHIRenderPipelineVK* m_boundPipeline = nullptr;

  struct StagedStreamBuffer {
    vk::Buffer buffer;
    vk::DeviceSize offset = 0;
  };
  std::vector<StagedStreamBuffer> m_pendingStreamBuffers; // index = stream slot

  struct StagedDescriptorWrite {
    uint32_t set = 0;
    vk::WriteDescriptorSet write{};
    // Owning storage for whatever the write points to.
    vk::DescriptorImageInfo imageInfo{};
    vk::DescriptorBufferInfo bufferInfo{};
  };
  // Keyed by (set, binding) so repeat loads to the same slot replace
  // instead of accumulate. Matches GL's stateful per-pipeline binding
  // semantics and bounds the per-draw push-descriptor work to the count
  // of distinct slots the shader actually declares.
  std::map<std::pair<uint32_t, uint32_t>, StagedDescriptorWrite> m_pendingDescriptorWrites;

  // For loadUniformBlockImmediate, we allocate a transient host-visible
  // buffer per call from a per-frame ring; this avoids needing push
  // constants for what GLSL declares as uniform blocks.
  struct TransientUniformAllocator {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    void* mapped = nullptr;
    vk::DeviceSize size = 0;
    vk::DeviceSize cursor = 0;
  };
  std::vector<TransientUniformAllocator> m_transientUniforms; // one per frame slot
  vk::DeviceSize m_transientUniformAlignment = 256;

  // Cached PhysicalDeviceLimits::timestampPeriod (nanoseconds per tick).
  // Populated lazily in setFrameSource alongside the UBO alignment cache.
  // 1.0 on this Tegra Orin's iGPU; left as a double for portability.
  double m_timestampPeriodNs = 1.0;

  // Default sampler used when loadTexture is called with sampler=nullptr.
  // The GL backend supports this by leaving texture-unit sampler state to
  // the texture's own glTexParameter values; Vulkan combined-image-samplers
  // require a real VkSampler handle, so we substitute a linear/repeat one.
  // Lazily created on first use because RHISamplerVK::create needs the
  // device, which isn't ready in the constructor.
  vk::UniqueSampler m_defaultSampler;
  vk::Sampler defaultSampler();

  // Reset the per-frame transient uniform cursor.
  void resetTransientUniforms();
  // Allocate `size` bytes from the current frame's transient ring; copy
  // `data` into it and return the (buffer, offset, size) triple for use
  // in a VkDescriptorBufferInfo. Asserts on overflow.
  struct TransientBufferAlloc {
    vk::Buffer buffer;
    vk::DeviceSize offset;
    vk::DeviceSize size;
  };
  TransientBufferAlloc allocTransientUniform(const void* data, size_t size);

  // Flush staged stream buffers + descriptor writes onto the current CB.
  void flushDrawState();

  // Stash a refcounted handle so it stays alive until the GPU has finished
  // executing the current frame's command buffer. Required for any
  // resource referenced from a recorded command — RHI callers may free
  // their own ptr between recording and submission.
  void keepAliveForFrame(boost::intrusive_ptr<RHIObject> obj);

  // Helper called by bindRenderPipeline to set every shader_object dynamic
  // state to a sensible default.
  void setDynamicStateDefaults();

  // -------- CUDA-VK interop sync ----------

  RHIVulkan::ExternalSemaphore m_interopTimeline;
  cudaExternalSemaphore_t m_interopTimelineCU = nullptr;

  // Single monotonic value allocator shared by both producers. Every signal consumes
  // the next value from here, so no two signals ever target the same timeline
  // value, and a given wait value can only be reached by its intended producer.
  uint64_t m_interopTimelineNextValue = 0;
  // Last value signaled by CUDA (set in signalCUDAToRHI). VK waits on this.
  uint64_t m_interopTimelineCudaSignaledValue = 0;
  // Last value signaled by VK (set in allocateVKSignalValue). CUDA (signalRHIToCUDA) waits on this.
  uint64_t m_interopTimelineVkSignaledValue = 0;
  // Allocate the next value for a VK signal. Drawn from the single shared counter.
  uint64_t interopAllocateVKSignalValue() { return (m_interopTimelineVkSignaledValue = ++m_interopTimelineNextValue); }

  // -------- Frame-completion timeline (host-waitable, internal) ----------

  // A non-exported timeline semaphore signaled at every swapBuffers/flush
  // submit with a monotonically increasing value. registerFrameCompletionFence()
  // hands out RHIFenceVK objects that host-wait on a target value, letting
  // another thread (e.g. the NvEnc submit worker) block until a frame's GPU
  // work completes. Because the semaphore is owned here for the device's
  // lifetime, a consumer that times out and drops its RHIFence can never
  // destroy a sync object a queue submission still references -- unlike a
  // per-frame VkFence. Created in setFrameSource.
  vk::UniqueSemaphore m_frameCompletionTimeline;
  // Last value allocated for the frame-completion timeline. Each swapBuffers/
  // flush submit signals ++m_frameCompletionValue; registerFrameCompletionFence
  // targets m_frameCompletionValue + 1 (the value the next submit will signal).
  // Touched only on the render thread.
  uint64_t m_frameCompletionValue = 0;


  // -------- debug: per-frame render-target dump --------
  //
  // When RHI_VK_DUMP_FRAME=<N> is set, every endRenderPass on the N-th
  // frame (0-indexed from process start) records a vkCmdCopyImageToBuffer
  // of the just-rendered image into a host-visible staging buffer. After
  // the frame's submit completes, each staging buffer is mapped, format-
  // converted to RGB(A)8, and written via stbi_write_png as
  //   <dir>/frame_<N>_pass_<P>_<window|offscreen>_<WxH>_<format>.png
  // <dir> defaults to /tmp/rhi-vk-dump (override via RHI_VK_DUMP_DIR).
  // Zero cost when the env var is unset (one int64 compare per endRenderPass).
  int64_t m_dumpFrameIndex = -1; // -1 disables
  std::string m_dumpDir;
  uint64_t m_dumpFrameCounter = 0; // incremented per swapBuffers
  uint32_t m_passCounterThisFrame = 0; // resets at swap/flush

  // Insert copyImageToBuffer + per-pass staging-buffer alloc into the
  // current CB; record the PendingDump on m_currentFrame for later write.
  // Called from endRenderPass when (m_dumpFrameCounter == m_dumpFrameIndex).
  void enqueuePassDump();

  // Submit a signal-only fence after the just-submitted frame CB and queue
  // a job for the background worker to wait on, then map and PNG-encode.
  // Returns immediately — the actual map / convert / stbi_write_png runs
  // off-thread so the render loop isn't stalled. Without this offload, a
  // multi-megapixel PNG encode (~50 ms) stretches the gap between submits
  // long enough to trip CUDA-side semaphore-acquire watchdogs on busy
  // pipelines (e.g. the mock-depth path).
  void processPendingDumps();

  // Background PNG-encoder worker.
  struct DumpJob {
    vk::UniqueFence fence; // signaled by an empty queue.submit after the frame CB
    std::vector<CurrentFrame::PendingDump> dumps;
  };
  std::thread m_dumpWorker;
  std::mutex m_dumpMutex;
  std::condition_variable m_dumpCv;
  std::deque<DumpJob> m_dumpQueue;
  bool m_dumpStop = false;
  void dumpWorkerThread();
};
