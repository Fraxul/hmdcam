#include "rhi/vk/RHIVK.h"
#include "rhi/vk/RHIBlendStateVK.h"
#include "rhi/vk/RHIBufferVK.h"
#include "rhi/vk/RHIDepthStencilStateVK.h"
#include "rhi/vk/RHIInteropBufferVK.h"
#include "rhi/vk/RHIInteropSurfaceVK.h"
#include "rhi/vk/RHIQueryVK.h"
#include "rhi/vk/RHIRenderPipelineVK.h"
#include "rhi/vk/RHIRenderTargetVK.h"
#include "rhi/vk/RHISamplerVK.h"
#include "rhi/vk/RHIShaderVK.h"
#include "rhi/vk/RHISurfaceVK.h"
#include "rhi/vk/RHIVKFrameSource.h"
#include "rhi/vk/RHIVulkan.h"
#include "rhi/vk/RHIWindowRenderTargetVK.h"
#include "rhi/cuda/CudaUtil.h"
#include "stb/stb_image_write.h"
#include <assert.h>
#include <cstring>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <system_error>

#define RHI_VK_NOT_IMPLEMENTED()                                \
  do {                                                          \
    fflush(stdout);                                             \
    fprintf(stderr, "RHIVK: not implemented: %s\n  at %s:%d\n", \
      __PRETTY_FUNCTION__, __FILE__, __LINE__);                 \
    abort();                                                    \
  } while (0)

RHIVK::RHIVK() {
  s_ndcZNearIsNegativeOne = false;
}

RHIVK::~RHIVK() {
  // Drain the dump worker before tearing down the device: its in-flight
  // jobs hold UniqueFence / UniqueBuffer / UniqueDeviceMemory handles
  // whose destructors call into the vk::Device. Stop-signal + join
  // guarantees the worker has popped every job from the queue.
  if (m_dumpWorker.joinable()) {
    {
      std::lock_guard<std::mutex> lock(m_dumpMutex);
      m_dumpStop = true;
    }
    m_dumpCv.notify_all();
    m_dumpWorker.join();
  }
  // Wait for any in-flight frames before tearing down command pool / image
  // views so the destruction order is safe.
  if (rhi() && rhi()->vk()) {
    rhi()->vk()->device().waitIdle();
  }

  if (m_interopTimelineCU) {
    cudaDestroyExternalSemaphore(m_interopTimelineCU);
    m_interopTimelineCU = nullptr;
  }
  if (m_rhiToCudaTimelineCU) {
    cudaDestroyExternalSemaphore(m_rhiToCudaTimelineCU);
    m_rhiToCudaTimelineCU = nullptr;
  }
}

void RHIVK::setFrameSource(VKFrameSource* source) {
  m_frameSource = source;
  VKFrameSource* frameSource = source;

  vk::Device device = rhi()->vk()->device();
  uint32_t queueFamily = rhi()->vk()->queueFamilyIndex();

  // Cache uniform-buffer offset alignment + timestamp period now that vk()
  // is live (both come off PhysicalDeviceLimits).
  {
    auto limits = rhi()->vk()->physicalDevice().getProperties().limits;
    m_transientUniformAlignment = std::max<vk::DeviceSize>(limits.minUniformBufferOffsetAlignment, 16);
    m_timestampPeriodNs = double(limits.timestampPeriod);
  }

  // Command pool: one per-buffer reset (we reset individual command buffers
  // between frames rather than the whole pool).
  vk::CommandPoolCreateInfo poolCi{vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily};
  m_commandPool = device.createCommandPoolUnique(poolCi);

  // One frame context per swap image. The ring holds the command buffer +
  // fence used to track that slot's in-flight work. We wait on the slot's
  // fence before recycling it.
  uint32_t frameCount = frameSource->swapImageCount();
  m_frameContexts.resize(frameCount);
  for (uint32_t i = 0; i < frameCount; ++i) {
    vk::CommandBufferAllocateInfo cbAi{m_commandPool.get(), vk::CommandBufferLevel::ePrimary, 1};
    auto buffers = device.allocateCommandBuffersUnique(cbAi);
    m_frameContexts[i].commandBuffer = std::move(buffers[0]);

    // Create fences in the SIGNALED state so the first frame's wait doesn't
    // block on a never-submitted fence.
    vk::FenceCreateInfo fenceCi{vk::FenceCreateFlagBits::eSignaled};
    m_frameContexts[i].frameFence = device.createFenceUnique(fenceCi);
  }
  m_frameContextIndex = 0;

  // One transient uniform ring per frame slot. 1 MiB is far more than
  // anything the engine packs into immediate uniform blocks today
  // (typical block is a vec4-sized push of camera params).
  constexpr vk::DeviceSize kTransientUniformRingSize = 1024 * 1024;
  m_transientUniforms.resize(frameCount);
  for (uint32_t i = 0; i < frameCount; ++i) {
    vk::BufferCreateInfo bci{
      vk::BufferCreateFlags(), kTransientUniformRingSize,
      vk::BufferUsageFlagBits::eUniformBuffer, vk::SharingMode::eExclusive};
    m_transientUniforms[i].buffer = device.createBufferUnique(bci);
    vk::MemoryRequirements req = device.getBufferMemoryRequirements(m_transientUniforms[i].buffer.get());
    vk::MemoryAllocateInfo mai{req.size,
      rhi()->vk()->findMemoryType(req.memoryTypeBits,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)};
    m_transientUniforms[i].memory = device.allocateMemoryUnique(mai);
    device.bindBufferMemory(m_transientUniforms[i].buffer.get(), m_transientUniforms[i].memory.get(), 0);
    m_transientUniforms[i].mapped = device.mapMemory(m_transientUniforms[i].memory.get(), 0, req.size);
    m_transientUniforms[i].size = req.size;
    m_transientUniforms[i].cursor = 0;
  }

  printf("RHIVK::setFrameSource: %u frame slots ready (transient uniform ring=%zu bytes/slot, ubo align=%zu)\n",
    frameCount, size_t(kTransientUniformRingSize), size_t(m_transientUniformAlignment));

  // Debug: RHI_VK_DUMP_FRAME=<N> picks a single frame to dump every
  // render-target pass to PNG. Lets the user diff stages while chasing
  // coordinate-convention bugs (Y-flip, etc.). Cost is zero when unset.
  if (const char* s = getenv("RHI_VK_DUMP_FRAME")) {
    m_dumpFrameIndex = atoll(s);
    m_dumpDir = getenv("RHI_VK_DUMP_DIR") ? getenv("RHI_VK_DUMP_DIR") : "/tmp/rhi-vk-dump";
    std::error_code ec;
    std::filesystem::create_directories(m_dumpDir, ec);
    if (ec) {
      fprintf(stderr, "RHIVK: warning: could not create dump dir %s: %s\n",
        m_dumpDir.c_str(), ec.message().c_str());
    }
    printf("RHIVK: will dump frame %lld render-target passes to %s\n",
      (long long) m_dumpFrameIndex, m_dumpDir.c_str());
  }

  // Frame-completion timeline: a plain (non-exported) timeline semaphore that
  // every swapBuffers/flush submit signals with an incrementing value. Drives
  // registerFrameCompletionFence (host-side cross-thread "wait for this frame").
  {
    vk::SemaphoreTypeCreateInfo stci{vk::SemaphoreType::eTimeline, /*initialValue=*/ 0};
    vk::SemaphoreCreateInfo sci{};
    sci.setPNext(&stci);
    m_frameCompletionTimeline = device.createSemaphoreUnique(sci);
    m_frameCompletionValue = 0;
  }

  // Build interop sync. Two separate single-writer timelines (see header): one
  // for CUDA->VK (CUDA signals, VK waits) and one for VK->CUDA (VK signals, CUDA
  // waits). Each is exported to CUDA. Keeping the directions on distinct
  // timelines is what keeps each one reliably monotonic.

  // cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd tells CUDA the FD points
  // at a timeline semaphore -- required for the params.fence.value field on
  // Signal/WaitExternalSemaphoresAsync to mean anything. cudaImportExternalSemaphore
  // takes ownership of the FD, so we null out our copy to avoid a double-close.
  auto importToCuda = [](RHIVulkan::ExternalSemaphore& sem, cudaExternalSemaphore_t* outCu) {
    cudaExternalSemaphoreHandleDesc desc = {};
    desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    desc.handle.fd = sem.fd;
    CUDA_CHECK(cudaImportExternalSemaphore(outCu, &desc));
    sem.fd = -1;
  };

  m_interopTimeline = rhi()->vk()->createExternalSemaphore(vk::SemaphoreType::eTimeline);
  importToCuda(m_interopTimeline, &m_interopTimelineCU);

  m_rhiToCudaTimeline = rhi()->vk()->createExternalSemaphore(vk::SemaphoreType::eTimeline);
  importToCuda(m_rhiToCudaTimeline, &m_rhiToCudaTimelineCU);
}

void RHIVK::invalidateSwapchainResources() {
  // Caller must have waitIdle'd the device. Wipe the image-view cache —
  // each entry references a VkImage that the driver is about to destroy
  // (or already has).
  m_swapImageViews.clear();
}

vk::ImageView RHIVK::swapImageViewFor(vk::Image image, vk::Format format) {
  auto it = m_swapImageViews.find(static_cast<VkImage>(image));
  if (it != m_swapImageViews.end())
    return it->second.get();

  vk::ImageViewCreateInfo viewCi{
    vk::ImageViewCreateFlags(),
    image,
    vk::ImageViewType::e2D,
    format,
    vk::ComponentMapping{ },
    vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };
  vk::UniqueImageView view = rhi()->vk()->device().createImageViewUnique(viewCi);
  vk::ImageView raw = view.get();
  m_swapImageViews[static_cast<VkImage>(image)] = std::move(view);
  return raw;
}

RHIDepthStencilState::ptr RHIVK::compileDepthStencilState(const RHIDepthStencilStateDescriptor& d) {
  return RHIDepthStencilState::ptr(new RHIDepthStencilStateVK(d));
}
RHIRenderTarget::ptr RHIVK::compileRenderTarget(const RHIRenderTargetDescriptor& d) {
  return RHIRenderTargetVK::compile(d);
}
RHISampler::ptr RHIVK::compileSampler(const RHISamplerDescriptor& d) {
  return RHISamplerVK::create(d);
}
RHIBlendState::ptr RHIVK::compileBlendState(const RHIBlendStateDescriptor& d) {
  return RHIBlendState::ptr(new RHIBlendStateVK(d));
}

RHIBuffer::ptr RHIVK::newBufferWithContents(const void* data, size_t size, RHIBufferUsageMode mode) {
  return RHIBufferVK::create(size, mode, data);
}
RHIBuffer::ptr RHIVK::newEmptyBuffer(size_t size, RHIBufferUsageMode mode) {
  return RHIBufferVK::create(size, mode, /*initialContents=*/ nullptr);
}
RHIBuffer::ptr RHIVK::newUniformBufferWithContents(const void* data, size_t size) {
  return RHIBufferVK::create(size, kBufferUsageCPUWriteOnly, data);
}
void RHIVK::loadBufferData(RHIBuffer::ptr buf, const void* data, size_t offset, size_t length) {
  if (length == 0) length = buf->size() - offset;
  static_cast<RHIBufferVK*>(buf.get())->loadData(data, length, offset);
}

RHISurface::ptr RHIVK::newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor& d) {
  return RHISurfaceVK::newTexture2D(width, height, d);
}
RHISurface::ptr RHIVK::newTexture3D(uint32_t, uint32_t, uint32_t, const RHISurfaceDescriptor&) { RHI_VK_NOT_IMPLEMENTED(); }
RHISurface::ptr RHIVK::newRenderbuffer2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor& d) {
  // VK doesn't distinguish renderbuffer vs texture — same underlying VkImage.
  return RHISurfaceVK::newTexture2D(width, height, d);
}
RHISurface::ptr RHIVK::newHMDSwapTexture(uint32_t, uint32_t, const RHISurfaceDescriptor&) { RHI_VK_NOT_IMPLEMENTED(); }
void RHIVK::loadTextureData(RHISurface::ptr texture, RHIVertexElementType /*sourceDataFormat*/, const void* sourceData) {
  // The contract set by the GL RHI assumes source data layout matches the surface format
  // (tight-packed, no per-element conversion).
  RHISurfaceVK* s = static_cast<RHISurfaceVK*>(texture.get());
  size_t bytes = static_cast<size_t>(s->width()) * s->height() * rhiSurfaceFormatSize(s->format());
  s->loadFromCPU(sourceData, bytes);
}
void RHIVK::generateTextureMips(RHISurface::ptr) { RHI_VK_NOT_IMPLEMENTED(); }
void RHIVK::readbackTexture(RHISurface::ptr, uint8_t, RHIVertexElementType, void*) { RHI_VK_NOT_IMPLEMENTED(); }
void RHIVK::fillOpenVRTextureStruct(RHISurface::ptr, vr::Texture_t*) { RHI_VK_NOT_IMPLEMENTED(); }

void RHIVK::setClearColor(const glm::vec4 color) {
  m_clearColor = color;
}
void RHIVK::setClearDepth(float v) { m_clearDepth = v; }
void RHIVK::setClearStencil(uint8_t v) { m_clearStencil = v; }

void RHIVK::ensureFrameCBOpen() {
  if (m_currentFrame.cbOpen)
    return;
  vk::Device device = rhi()->vk()->device();
  FrameContext& fc = m_frameContexts[m_frameContextIndex];
  // Wait on this slot's previous frame's GPU work. Guarantees the CB +
  // any keepAlive resources are safe to recycle.
  (void) device.waitForFences(fc.frameFence.get(), VK_TRUE, std::numeric_limits<uint64_t>::max());
  device.resetFences(fc.frameFence.get());
  resetTransientUniforms();
  fc.keepAlive.clear();
  fc.commandBuffer->reset();
  fc.commandBuffer->begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  m_currentFrame.cbOpen = true;
  m_currentFrame.commandBuffer = fc.commandBuffer.get();
  m_currentFrame.frameFence = fc.frameFence.get();
}

namespace {

// Map an RHI load action to a Vulkan attachment load op. kLoadInvalidate
// matches GL's glDiscardFramebufferEXT (contents become undefined) → eDontCare;
// kLoadPreserveContents / kLoadUnspecified keep prior contents → eLoad.
vk::AttachmentLoadOp vkLoadOpFor(RHIRenderTargetLoadAction action) {
  switch (action) {
    case kLoadClear: return vk::AttachmentLoadOp::eClear;
    case kLoadInvalidate: return vk::AttachmentLoadOp::eDontCare;
    default: return vk::AttachmentLoadOp::eLoad;
  }
}

} // namespace

void RHIVK::beginRenderPass(RHIRenderTarget::ptr target,
  RHIRenderTargetLoadAction colorLoadAction,
  RHIRenderTargetLoadAction depthLoadAction,
  RHIRenderTargetLoadAction stencilLoadAction) {
  if (m_currentFrame.passActive) {
    fprintf(stderr, "RHIVK::beginRenderPass: previous render pass not closed\n");
    abort();
  }

  // Mirror the GL backend's load-action defaulting: depth follows color and
  // stencil follows depth when left unspecified.
  if (depthLoadAction == kLoadUnspecified)
    depthLoadAction = colorLoadAction;
  if (stencilLoadAction == kLoadUnspecified)
    stencilLoadAction = depthLoadAction;

  ensureFrameCBOpen();

  // Reset per-pass attachment bookkeeping.
  m_currentFrame.passColorAttachments.clear();
  m_currentFrame.passHasDepthStencil = false;
  m_currentFrame.passDepthStencilHasDepth = false;
  m_currentFrame.passDepthStencilHasStencil = false;
  m_currentFrame.passDepthStencilImage = nullptr;
  m_currentFrame.passDepthStencilView = nullptr;

  bool isWindow = target->isWindowRenderTarget();
  if (isWindow) {
    if (m_currentFrame.hasWindowImage) {
      fprintf(stderr, "RHIVK::beginRenderPass: multiple window-RT passes in one frame not supported\n");
      abort();
    }
    assert(m_frameSource);
    // Acquire the next swap image (may block on host until available).
    // The swap image is per-pass; the slot fence we waited on in
    // ensureFrameCBOpen guards CB recycling, separately.
    VKFrameInfo frame = m_frameSource->acquireVKFrame();
    m_currentFrame.hasWindowImage = true;
    m_currentFrame.swapchainIndex = frame.swapchainIndex;
    m_currentFrame.swapImage = frame.swapchainImage;
    m_currentFrame.swapExtent = frame.extent;
    m_currentFrame.swapFormat = frame.format;
    m_currentFrame.imageAcquired = frame.imageAcquired;
    m_currentFrame.renderFinished = frame.renderFinished;

    CurrentFrame::PassColorAttachment att;
    att.image = frame.swapchainImage;
    att.view = swapImageViewFor(frame.swapchainImage, frame.format);
    att.format = frame.format;
    // Window RTs need to end up in ePresentSrcKHR
    att.targetLayout = vk::ImageLayout::ePresentSrcKHR;
    m_currentFrame.passColorAttachments.push_back(att);
    m_currentFrame.passExtent = frame.extent;
  } else {
    // Offscreen RT: zero or more color attachments + an optional
    // depth-stencil attachment, matching the GL backend.
    auto* offRT = static_cast<RHIRenderTargetVK*>(target.get());
    if (offRT->colorTargetCount() == 0 && !offRT->hasDepthStencilTarget()) {
      fprintf(stderr, "RHIVK::beginRenderPass: offscreen RT has no attachments\n");
      abort();
    }
    m_currentFrame.passExtent = vk::Extent2D{offRT->width(), offRT->height()};
    for (size_t i = 0; i < offRT->colorTargetCount(); ++i) {
      RHISurfaceVK* c = offRT->colorAttachment(i);
      CurrentFrame::PassColorAttachment att;
      att.image = c->vkImage();
      att.view = c->vkImageView();
      att.format = c->vkFormat();
      // Offscreen surfaces can specify their desired at-rest image layout.
      // Typically, this will be eShaderReadOnlyOptimal.
      att.targetLayout = c->vkDesiredImageLayout();
      m_currentFrame.passColorAttachments.push_back(att);
    }
    if (offRT->hasDepthStencilTarget()) {
      RHISurfaceVK* ds = offRT->depthStencilAttachment();
      m_currentFrame.passHasDepthStencil = true;
      m_currentFrame.passDepthStencilHasDepth = rhiSurfaceFormatHasDepth(ds->format());
      m_currentFrame.passDepthStencilHasStencil = rhiSurfaceFormatHasStencil(ds->format());
      m_currentFrame.passDepthStencilImage = ds->vkImage();
      m_currentFrame.passDepthStencilView = ds->vkImageView();
    }
  }

  // Primary color alias (used by the dump path + viewport/scissor defaults).
  if (!m_currentFrame.passColorAttachments.empty()) {
    m_currentFrame.passImage = m_currentFrame.passColorAttachments[0].image;
    m_currentFrame.passImageView = m_currentFrame.passColorAttachments[0].view;
    m_currentFrame.passFormat = m_currentFrame.passColorAttachments[0].format;
    m_currentFrame.passImageLayout = m_currentFrame.passColorAttachments[0].targetLayout;
  } else {
    m_currentFrame.passImage = nullptr;
    m_currentFrame.passImageView = nullptr;
    m_currentFrame.passFormat = vk::Format{};
    m_currentFrame.passImageLayout = vk::ImageLayout{};
  }

  m_currentFrame.passActive = true;
  m_currentFrame.passIsWindowRT = isWindow;

  vk::CommandBuffer cb = m_currentFrame.commandBuffer;

  // Layout transitions: UNDEFINED → *_ATTACHMENT_OPTIMAL for every attachment.
  // UNDEFINED discards prior contents, which is correct for clear/invalidate.
  // (Full cross-pass layout tracking — needed to truly preserve contents on
  // kLoadPreserveContents — is still deferred; this matches the prior
  // color-only behavior.)
  std::vector<vk::ImageMemoryBarrier> toAttachment;
  toAttachment.reserve(m_currentFrame.passColorAttachments.size() + 1);
  for (const auto& att : m_currentFrame.passColorAttachments) {
    toAttachment.push_back(vk::ImageMemoryBarrier{
      vk::AccessFlags(),
      vk::AccessFlagBits::eColorAttachmentWrite,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eColorAttachmentOptimal,
      VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED,
      att.image,
      vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
  }
  vk::PipelineStageFlags toAttachmentDstStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  if (m_currentFrame.passHasDepthStencil) {
    vk::ImageAspectFlags aspect;
    if (m_currentFrame.passDepthStencilHasDepth) aspect |= vk::ImageAspectFlagBits::eDepth;
    if (m_currentFrame.passDepthStencilHasStencil) aspect |= vk::ImageAspectFlagBits::eStencil;
    toAttachment.push_back(vk::ImageMemoryBarrier{
      vk::AccessFlags(),
      vk::AccessFlagBits::eDepthStencilAttachmentWrite,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eDepthStencilAttachmentOptimal,
      VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED,
      m_currentFrame.passDepthStencilImage,
      vk::ImageSubresourceRange{aspect, 0, 1, 0, 1},
    });
    toAttachmentDstStage |= vk::PipelineStageFlagBits::eEarlyFragmentTests;
  }
  // For the window swap image, the begin barrier must form an execution
  // dependency with the imageAcquired semaphore (waited at
  // COLOR_ATTACHMENT_OUTPUT in swapBuffers). srcStage = TOP_OF_PIPE would let
  // the UNDEFINED→COLOR transition race the presentation engine's read of the
  // image (WRITE_AFTER_READ hazard), so match the semaphore's wait stage.
  // Offscreen attachments aren't acquire-dependent, so TOP_OF_PIPE is fine.
  vk::PipelineStageFlags toAttachmentSrcStage = isWindow
    ? vk::PipelineStageFlagBits::eColorAttachmentOutput
    : vk::PipelineStageFlagBits::eTopOfPipe;
  if (!toAttachment.empty()) {
    cb.pipelineBarrier(
      toAttachmentSrcStage,
      toAttachmentDstStage,
      vk::DependencyFlags(), 0, nullptr, 0, nullptr,
      uint32_t(toAttachment.size()), toAttachment.data());
  }

  // Color attachment infos (clear value from setClearColor).
  vk::ClearValue colorClear{};
  colorClear.color.float32[0] = m_clearColor.r;
  colorClear.color.float32[1] = m_clearColor.g;
  colorClear.color.float32[2] = m_clearColor.b;
  colorClear.color.float32[3] = m_clearColor.a;
  std::vector<vk::RenderingAttachmentInfoKHR> colorInfos;
  colorInfos.reserve(m_currentFrame.passColorAttachments.size());
  for (const auto& att : m_currentFrame.passColorAttachments) {
    colorInfos.push_back(vk::RenderingAttachmentInfoKHR{
      att.view,
      vk::ImageLayout::eColorAttachmentOptimal,
      vk::ResolveModeFlagBits::eNone,
      {},
      vk::ImageLayout::eUndefined,
      vkLoadOpFor(colorLoadAction),
      vk::AttachmentStoreOp::eStore,
      colorClear,
    });
  }

  // Depth + stencil attachment infos (clear value from setClearDepth/Stencil).
  // A combined depth-stencil format points both attachments at the same view,
  // each carrying its own load op.
  vk::ClearValue depthStencilClear{};
  depthStencilClear.depthStencil.depth = m_clearDepth;
  depthStencilClear.depthStencil.stencil = m_clearStencil;
  vk::RenderingAttachmentInfoKHR depthInfo{
    m_currentFrame.passDepthStencilView,
    vk::ImageLayout::eDepthStencilAttachmentOptimal,
    vk::ResolveModeFlagBits::eNone,
    {},
    vk::ImageLayout::eUndefined,
    vkLoadOpFor(depthLoadAction),
    vk::AttachmentStoreOp::eStore,
    depthStencilClear,
  };
  vk::RenderingAttachmentInfoKHR stencilInfo{
    m_currentFrame.passDepthStencilView,
    vk::ImageLayout::eDepthStencilAttachmentOptimal,
    vk::ResolveModeFlagBits::eNone,
    {},
    vk::ImageLayout::eUndefined,
    vkLoadOpFor(stencilLoadAction),
    vk::AttachmentStoreOp::eStore,
    depthStencilClear,
  };
  bool useDepth = m_currentFrame.passHasDepthStencil && m_currentFrame.passDepthStencilHasDepth;
  bool useStencil = m_currentFrame.passHasDepthStencil && m_currentFrame.passDepthStencilHasStencil;

  vk::RenderingInfoKHR renderingInfo{
    vk::RenderingFlags(),
    vk::Rect2D{vk::Offset2D{0, 0}, m_currentFrame.passExtent},
    1, // layerCount
    0, // viewMask
    uint32_t(colorInfos.size()),
    colorInfos.empty() ? nullptr : colorInfos.data(),
    useDepth ? &depthInfo : nullptr,
    useStencil ? &stencilInfo : nullptr,
  };

  cb.beginRenderingKHR(renderingInfo);

  // shader_object requires every dynamic state to be set before the first
  // draw. Seed all of them to known defaults here so a subsequent draw
  // succeeds even if the caller binds no state. User bindBlendState /
  // bindDepthStencilState / setCullState / setViewport / setScissorRect
  // overrides individual pieces; bindRenderPipeline only writes
  // pipeline-specific dynamic state (vertex input, topology, etc.).
  setDynamicStateDefaults();
}

void RHIVK::endRenderPass(RHIRenderTarget::ptr /*target*/) {
  if (!m_currentFrame.passActive) {
    fprintf(stderr, "RHIVK::endRenderPass: no render pass in progress\n");
    abort();
  }
  vk::CommandBuffer cb = m_currentFrame.commandBuffer;
  cb.endRenderingKHR();

  // Destination stage and access is set for non-window RTs so that they can
  // be sampled in subsequent fragment shaders. For window RTs, eBottomOfPipe
  // guarantees that everything's done before handoff to the presentation engine.
  vk::AccessFlags colorDstAccess = m_currentFrame.passIsWindowRT
    ? vk::AccessFlags()
    : vk::AccessFlagBits::eShaderRead;
  vk::PipelineStageFlags dstStage = m_currentFrame.passIsWindowRT
    ? vk::PipelineStageFlagBits::eBottomOfPipe
    : vk::PipelineStageFlagBits::eFragmentShader;

  std::vector<vk::ImageMemoryBarrier> toFinal;
  toFinal.reserve(m_currentFrame.passColorAttachments.size() + 1);
  for (const auto& att : m_currentFrame.passColorAttachments) {
    toFinal.push_back(vk::ImageMemoryBarrier{
      vk::AccessFlagBits::eColorAttachmentWrite,
      colorDstAccess,
      vk::ImageLayout::eColorAttachmentOptimal,
      att.targetLayout,
      VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED,
      att.image,
      vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
    });
  }
  vk::PipelineStageFlags srcStage{};
  if (!m_currentFrame.passColorAttachments.empty())
    srcStage |= vk::PipelineStageFlagBits::eColorAttachmentOutput;
  if (m_currentFrame.passHasDepthStencil) {
    // Offscreen depth-stencil transitions to shader-read so a later pass can
    // sample it (e.g. as a depth texture). Offscreen-only: window RTs never
    // carry a depth attachment.
    vk::ImageAspectFlags aspect;
    if (m_currentFrame.passDepthStencilHasDepth) aspect |= vk::ImageAspectFlagBits::eDepth;
    if (m_currentFrame.passDepthStencilHasStencil) aspect |= vk::ImageAspectFlagBits::eStencil;
    toFinal.push_back(vk::ImageMemoryBarrier{
      vk::AccessFlagBits::eDepthStencilAttachmentWrite,
      vk::AccessFlagBits::eShaderRead,
      vk::ImageLayout::eDepthStencilAttachmentOptimal,
      vk::ImageLayout::eShaderReadOnlyOptimal,
      VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED,
      m_currentFrame.passDepthStencilImage,
      vk::ImageSubresourceRange{aspect, 0, 1, 0, 1},
    });
    srcStage |= vk::PipelineStageFlagBits::eLateFragmentTests;
  }
  if (!toFinal.empty()) {
    cb.pipelineBarrier(
      srcStage,
      dstStage,
      vk::DependencyFlags(), 0, nullptr, 0, nullptr,
      uint32_t(toFinal.size()), toFinal.data());
  }

  // Debug: optionally snapshot this pass for PNG dump. Gated by env var; runs
  // only on the configured frame. The dump path snapshots the primary color
  // attachment, so skip it for a depth-only pass (no color image to copy).
  if (m_currentFrame.passImage && static_cast<int64_t>(m_dumpFrameCounter) == m_dumpFrameIndex) {
    enqueuePassDump();
  }
  ++m_passCounterThisFrame;

  // Pass scope done. The CB stays open for the next begin* (next render
  // pass, timer query, etc.); swapBuffers / flush handles the submit.
  m_currentFrame.passActive = false;
  m_currentFrame.passIsWindowRT = false;
  m_currentFrame.passColorAttachments.clear();
  m_currentFrame.passHasDepthStencil = false;
  m_currentFrame.passDepthStencilHasDepth = false;
  m_currentFrame.passDepthStencilHasStencil = false;
  m_currentFrame.passDepthStencilImage = nullptr;
  m_currentFrame.passDepthStencilView = nullptr;
  m_currentFrame.passImage = nullptr;
  m_currentFrame.passImageView = nullptr;
  m_currentFrame.passExtent = vk::Extent2D{};
  m_currentFrame.passFormat = vk::Format{};
  m_boundPipeline = nullptr;
  m_pendingStreamBuffers.clear();
  m_pendingDescriptorWrites.clear();
}
void RHIVK::beginComputePass() { RHI_VK_NOT_IMPLEMENTED(); }
void RHIVK::endComputePass() { RHI_VK_NOT_IMPLEMENTED(); }

void RHIVK::blitTex(RHISurface::ptr, uint8_t, RHIRect, RHIRect) { RHI_VK_NOT_IMPLEMENTED(); }

// -------- Helper conversions --------

namespace {

vk::CompareOp vkCompareOpFor(RHIDepthStencilCompareFunction f) {
  switch (f) {
    case kCompareNever: return vk::CompareOp::eNever;
    case kCompareLess: return vk::CompareOp::eLess;
    case kCompareLessEqual: return vk::CompareOp::eLessOrEqual;
    case kCompareEqual: return vk::CompareOp::eEqual;
    case kCompareNotEqual: return vk::CompareOp::eNotEqual;
    case kCompareGreaterEqual: return vk::CompareOp::eGreaterOrEqual;
    case kCompareGreater: return vk::CompareOp::eGreater;
    case kCompareAlways: return vk::CompareOp::eAlways;
  }
  return vk::CompareOp::eAlways;
}

vk::StencilOp vkStencilOpFor(RHIStencilOperation op) {
  switch (op) {
    case kStencilKeep: return vk::StencilOp::eKeep;
    case kStencilZero: return vk::StencilOp::eZero;
    case kStencilReplace: return vk::StencilOp::eReplace;
    case kStencilIncrementClamp: return vk::StencilOp::eIncrementAndClamp;
    case kStencilDecrementClamp: return vk::StencilOp::eDecrementAndClamp;
    case kStencilInvert: return vk::StencilOp::eInvert;
    case kStencilIncrementWrap: return vk::StencilOp::eIncrementAndWrap;
    case kStencilDecrementWrap: return vk::StencilOp::eDecrementAndWrap;
  }
  return vk::StencilOp::eKeep;
}

vk::BlendFactor vkBlendFactorFor(RHIBlendWeight w) {
  switch (w) {
    case kBlendZero: return vk::BlendFactor::eZero;
    case kBlendOne: return vk::BlendFactor::eOne;
    case kBlendSourceColor: return vk::BlendFactor::eSrcColor;
    case kBlendOneMinusSourceColor: return vk::BlendFactor::eOneMinusSrcColor;
    case kBlendDestColor: return vk::BlendFactor::eDstColor;
    case kBlendOneMinusDestColor: return vk::BlendFactor::eOneMinusDstColor;
    case kBlendSourceAlpha: return vk::BlendFactor::eSrcAlpha;
    case kBlendOneMinusSourceAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
    case kBlendDestAlpha: return vk::BlendFactor::eDstAlpha;
    case kBlendOneMinusDestAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
    case kBlendConstantColor: return vk::BlendFactor::eConstantColor;
    case kBlendOneMinusConstantColor: return vk::BlendFactor::eOneMinusConstantColor;
    case kBlendConstantAlpha: return vk::BlendFactor::eConstantAlpha;
    case kBlendOneMinusConstantAlpha: return vk::BlendFactor::eOneMinusConstantAlpha;
    case kBlendSourceAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
    case kBlendSource1Color: return vk::BlendFactor::eSrc1Color;
    case kBlendOneMinusSource1Color: return vk::BlendFactor::eOneMinusSrc1Color;
    case kBlendSource1Alpha: return vk::BlendFactor::eSrc1Alpha;
    case kBlendOneMinusSource1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
  }
  return vk::BlendFactor::eZero;
}

vk::BlendOp vkBlendOpFor(RHIBlendFunc f) {
  switch (f) {
    case kBlendFuncAdd: return vk::BlendOp::eAdd;
    case kBlendFuncSubtract: return vk::BlendOp::eSubtract;
    case kBlendFuncReverseSubtract: return vk::BlendOp::eReverseSubtract;
    case kBlendFuncMin: return vk::BlendOp::eMin;
    case kBlendFuncMax: return vk::BlendOp::eMax;
  }
  return vk::BlendOp::eAdd;
}

} // namespace

void RHIVK::setViewport(const RHIRect& r) {
  if (!m_currentFrame.passActive) return;
  vk::Viewport vp{
    float(r.x), float(r.y), float(r.width), float(r.height),
    0.0f, 1.0f};
  vk::CommandBuffer cb = m_currentFrame.commandBuffer;
  cb.setViewportWithCountEXT(1, &vp);
  // shader_object requires scissorCount == viewportCount at draw time. The
  // engine leaves scissoring effectively disabled here (the GL path disabled
  // GL_SCISSOR_TEST per pass), so cover the whole render target.
  vk::Rect2D scissor{
    {0, 0},
    m_currentFrame.passExtent
  };
  cb.setScissorWithCountEXT(1, &scissor);
  m_currentFrame.viewportCount = 1;
}
void RHIVK::setViewports(const RHIRect* rects, size_t count) {
  if (!m_currentFrame.passActive || count == 0) return;
  std::vector<vk::Viewport> vps(count);
  std::vector<vk::Rect2D> scissors(count);
  vk::Rect2D fullRT{
    {0, 0},
    m_currentFrame.passExtent
  };
  for (size_t i = 0; i < count; ++i) {
    vps[i] = vk::Viewport{
      float(rects[i].x), float(rects[i].y), float(rects[i].width), float(rects[i].height),
      0.0f, 1.0f};
    // Matching full-RT scissor per viewport (see setViewport).
    scissors[i] = fullRT;
  }
  vk::CommandBuffer cb = m_currentFrame.commandBuffer;
  cb.setViewportWithCountEXT(uint32_t(count), vps.data());
  cb.setScissorWithCountEXT(uint32_t(count), scissors.data());
  m_currentFrame.viewportCount = uint32_t(count);
}
void RHIVK::setDepthBias(float slopeScale, float constantBias) {
  if (!m_currentFrame.passActive) return;
  m_currentFrame.commandBuffer.setDepthBiasEnableEXT(VK_TRUE);
  m_currentFrame.commandBuffer.setDepthBias(constantBias, /*clamp=*/ 0.0f, slopeScale);
}

void RHIVK::bindRenderPipeline(RHIRenderPipeline::ptr pipeline) {
  if (!m_currentFrame.passActive) {
    fprintf(stderr, "RHIVK::bindRenderPipeline: no render pass active\n");
    abort();
  }
  RHIRenderPipelineVK* p = static_cast<RHIRenderPipelineVK*>(pipeline.get());
  m_boundPipeline = p;

  vk::CommandBuffer cb = m_currentFrame.commandBuffer;

  // Bind all stages in one call so the driver knows the full program.
  // Stages not declared (e.g. geometry) get a null shader handle in the
  // appropriate slot (we bind a fixed set of stage slots so unbound ones
  // explicitly clear prior bindings).
  static constexpr vk::ShaderStageFlagBits kAllStages[] = {
    vk::ShaderStageFlagBits::eVertex,
    vk::ShaderStageFlagBits::eTessellationControl,
    vk::ShaderStageFlagBits::eTessellationEvaluation,
    vk::ShaderStageFlagBits::eGeometry,
    vk::ShaderStageFlagBits::eFragment,
  };
  std::vector<vk::ShaderEXT> shaderHandles(std::size(kAllStages), vk::ShaderEXT{});
  const auto& stages = p->shaderStages();
  const auto& shaders = p->shaderObjects();
  for (size_t i = 0; i < stages.size(); ++i) {
    for (size_t j = 0; j < std::size(kAllStages); ++j) {
      if (stages[i] == kAllStages[j]) {
        shaderHandles[j] = shaders[i];
        break;
      }
    }
  }
  cb.bindShadersEXT(uint32_t(std::size(kAllStages)), kAllStages, shaderHandles.data());

  // Pipeline-derived dynamic state. The non-pipeline-derived defaults
  // (blend/depth/stencil/raster) are seeded once per render pass in
  // beginRenderPass and only overridden by user state binds — NOT by
  // bindRenderPipeline. The engine's pattern is to bind state then bind
  // pipeline expecting state to persist across pipeline binds (true in
  // GL where state is independent of pipeline; we preserve that here).
  cb.setVertexInputEXT(p->vertexBindings(), p->vertexAttributes());
  cb.setPrimitiveTopologyEXT(p->primitiveTopology());
  cb.setPrimitiveRestartEnableEXT(m_boundPipeline->descriptor().primitiveRestartEnabled ? VK_TRUE : VK_FALSE);
  cb.setRasterizerDiscardEnableEXT(m_boundPipeline->descriptor().rasterizationEnabled ? VK_FALSE : VK_TRUE);
  cb.setAlphaToCoverageEnableEXT(m_boundPipeline->descriptor().alphaToCoverageEnabled ? VK_TRUE : VK_FALSE);

  // Clear pending bind-time state so the next frame starts clean.
  m_pendingStreamBuffers.clear();
  m_pendingDescriptorWrites.clear();
}

void RHIVK::setDynamicStateDefaults() {
  vk::CommandBuffer cb = m_currentFrame.commandBuffer;
  // Rasterization
  cb.setRasterizerDiscardEnableEXT(VK_FALSE);
  cb.setPolygonModeEXT(vk::PolygonMode::eFill);
  cb.setCullModeEXT(vk::CullModeFlagBits::eNone);
  cb.setFrontFaceEXT(vk::FrontFace::eCounterClockwise);
  cb.setDepthClampEnableEXT(VK_FALSE);
  cb.setDepthBiasEnableEXT(VK_FALSE);
  cb.setLineWidth(1.0f);
  cb.setRasterizationSamplesEXT(vk::SampleCountFlagBits::e1);
  const vk::SampleMask sampleMask = 0xffffffff;
  cb.setSampleMaskEXT(vk::SampleCountFlagBits::e1, &sampleMask);
  cb.setAlphaToCoverageEnableEXT(VK_FALSE);
  cb.setAlphaToOneEnableEXT(VK_FALSE);
  cb.setLogicOpEnableEXT(VK_FALSE);
  cb.setPrimitiveRestartEnableEXT(VK_FALSE);

  // Depth / stencil
  cb.setDepthTestEnableEXT(VK_FALSE);
  cb.setDepthWriteEnableEXT(VK_FALSE);
  cb.setDepthCompareOpEXT(vk::CompareOp::eLess);
  cb.setDepthBoundsTestEnableEXT(VK_FALSE);
  cb.setStencilTestEnableEXT(VK_FALSE);
  cb.setStencilOpEXT(vk::StencilFaceFlagBits::eFrontAndBack,
    vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways);
  cb.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack, 0xff);
  cb.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, 0xff);
  cb.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, 0);

  // Color blend — one entry per bound color attachment, blending disabled,
  // all channels written. shader_object requires these dynamic states to be
  // set for every attachment the pass uses.
  uint32_t colorCount = uint32_t(m_currentFrame.passColorAttachments.size());
  if (colorCount > 0) {
    const vk::ColorBlendEquationEXT blendEq{
      vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
      vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd};
    const vk::ColorComponentFlags writeAll =
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
      vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    std::vector<vk::Bool32> blendEnables(colorCount, VK_FALSE);
    std::vector<vk::ColorBlendEquationEXT> blendEqs(colorCount, blendEq);
    std::vector<vk::ColorComponentFlags> writeMasks(colorCount, writeAll);
    // const pointer disambiguates from the ArrayProxy overload (vk::Bool32 is
    // uint32_t, so colorCount could otherwise bind to a single-element proxy).
    cb.setColorBlendEnableEXT(0, colorCount, static_cast<const vk::Bool32*>(blendEnables.data()));
    cb.setColorBlendEquationEXT(0, colorCount, blendEqs.data());
    cb.setColorWriteMaskEXT(0, colorCount, writeMasks.data());
  }

  // Default viewport + scissor cover the whole render target.
  vk::Viewport vp{0.0f, 0.0f, float(m_currentFrame.passExtent.width), float(m_currentFrame.passExtent.height), 0.0f, 1.0f};
  cb.setViewportWithCountEXT(1, &vp);
  vk::Rect2D scissor{
    {0, 0},
    m_currentFrame.passExtent
  };
  cb.setScissorWithCountEXT(1, &scissor);
  m_currentFrame.viewportCount = 1;
}

void RHIVK::bindStreamBuffer(size_t streamIndex, RHIBuffer::ptr buffer, size_t baseOffsetBytes) {
  if (m_pendingStreamBuffers.size() <= streamIndex)
    m_pendingStreamBuffers.resize(streamIndex + 1);
  StagedStreamBuffer s;
  s.buffer = static_cast<RHIBufferVK*>(buffer.get())->vkBuffer();
  s.offset = baseOffsetBytes;
  m_pendingStreamBuffers[streamIndex] = s;
  static const bool s_trace = getenv("RHI_VK_TRACE_BINDS") && atoi(getenv("RHI_VK_TRACE_BINDS"));
  if (s_trace) {
    fprintf(stderr, "  bindStreamBuffer slot=%zu vkBuf=%p offset=%zu bufSize=%zu\n",
      streamIndex, (void*) (uintptr_t) (VkBuffer) s.buffer, baseOffsetBytes, buffer->size());
  }
  keepAliveForFrame(buffer);
}

void RHIVK::keepAliveForFrame(boost::intrusive_ptr<RHIObject> obj) {
  // Park the ref on the per-slot keepAlive list. Released the next time
  // this slot's CB is recycled, which happens after we wait on the slot
  // fence in ensureFrameCBOpen — so refs live at least until the GPU has
  // finished executing the CB that referenced them.
  if (!m_currentFrame.cbOpen) return;
  if (m_frameContextIndex >= m_frameContexts.size()) return;
  m_frameContexts[m_frameContextIndex].keepAlive.push_back(std::move(obj));
}

void RHIVK::bindDepthStencilState(RHIDepthStencilState::ptr state) {
  if (!m_currentFrame.passActive) return;
  const RHIDepthStencilStateDescriptor& d = static_cast<RHIDepthStencilStateVK*>(state.get())->descriptor();
  vk::CommandBuffer cb = m_currentFrame.commandBuffer;
  cb.setDepthTestEnableEXT(d.depthTestEnable ? VK_TRUE : VK_FALSE);
  cb.setDepthWriteEnableEXT(d.depthWriteEnable ? VK_TRUE : VK_FALSE);
  cb.setDepthCompareOpEXT(vkCompareOpFor(d.depthFunction));
  cb.setStencilTestEnableEXT(d.stencilTestEnable ? VK_TRUE : VK_FALSE);
  if (d.stencilTestEnable) {
    cb.setStencilOpEXT(vk::StencilFaceFlagBits::eFront,
      vkStencilOpFor(d.stencilFront.failOp),
      vkStencilOpFor(d.stencilFront.passOp),
      vkStencilOpFor(d.stencilFront.depthFailOp),
      vkCompareOpFor(d.stencilFront.compareFunc));
    cb.setStencilOpEXT(vk::StencilFaceFlagBits::eBack,
      vkStencilOpFor(d.stencilBack.failOp),
      vkStencilOpFor(d.stencilBack.passOp),
      vkStencilOpFor(d.stencilBack.depthFailOp),
      vkCompareOpFor(d.stencilBack.compareFunc));
    cb.setStencilReference(vk::StencilFaceFlagBits::eFront, d.stencilFront.referenceValue);
    cb.setStencilReference(vk::StencilFaceFlagBits::eBack, d.stencilBack.referenceValue);
    cb.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack, d.stencilMask);
    cb.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, d.stencilMask);
  }
}

void RHIVK::bindBlendState(RHIBlendState::ptr state) {
  if (!m_currentFrame.passActive) return;
  const RHIBlendStateDescriptor& d = static_cast<RHIBlendStateVK*>(state.get())->descriptor();
  vk::CommandBuffer cb = m_currentFrame.commandBuffer;

  uint32_t colorCount = uint32_t(m_currentFrame.passColorAttachments.size());
  if (colorCount == 0 || d.targetBlendStates.empty()) return;

  // The engine emits a single element to mean "same blend for every color
  // attachment"; honor per-attachment entries when supplied, otherwise
  // replicate element 0 across all bound color attachments.
  std::vector<vk::Bool32> enables(colorCount);
  std::vector<vk::ColorBlendEquationEXT> eqs(colorCount);
  for (uint32_t i = 0; i < colorCount; ++i) {
    const auto& el = d.targetBlendStates[i < d.targetBlendStates.size() ? i : 0];
    enables[i] = el.blendEnabled ? VK_TRUE : VK_FALSE;
    eqs[i] = vk::ColorBlendEquationEXT{
      vkBlendFactorFor(el.colorSource), vkBlendFactorFor(el.colorDest), vkBlendOpFor(el.colorFunc),
      vkBlendFactorFor(el.alphaSource), vkBlendFactorFor(el.alphaDest), vkBlendOpFor(el.alphaFunc)};
  }
  // const pointer disambiguates from the ArrayProxy overload (see note in
  // setDynamicStateDefaults).
  cb.setColorBlendEnableEXT(0, colorCount, static_cast<const vk::Bool32*>(enables.data()));
  cb.setColorBlendEquationEXT(0, colorCount, eqs.data());
  cb.setBlendConstants(&d.constantColor.r);
}

void RHIVK::setCullState(RHICullState s) {
  if (!m_currentFrame.passActive) return;
  vk::CullModeFlags mode;
  switch (s) {
    case kCullFrontFaces: mode = vk::CullModeFlagBits::eFront; break;
    case kCullBackFaces: mode = vk::CullModeFlagBits::eBack; break;
    default: mode = vk::CullModeFlagBits::eNone; break;
  }
  m_currentFrame.commandBuffer.setCullModeEXT(mode);
}

void RHIVK::setScissorRect(const RHIRect& r) {
  if (!m_currentFrame.passActive) return;
  vk::Rect2D scissor{
    {     int32_t(r.x),       int32_t(r.y)},
    {uint32_t(r.width), uint32_t(r.height)}
  };
  // Replicate across the current viewport count to satisfy shader_object's
  // scissorCount == viewportCount requirement (single-rect API → same rect
  // for every viewport).
  std::vector<vk::Rect2D> scissors(m_currentFrame.viewportCount, scissor);
  m_currentFrame.commandBuffer.setScissorWithCountEXT(m_currentFrame.viewportCount, scissors.data());
}
void RHIVK::clearScissorRect() {
  if (!m_currentFrame.passActive) return;
  vk::Rect2D scissor{
    {0, 0},
    m_currentFrame.passExtent
  };
  std::vector<vk::Rect2D> scissors(m_currentFrame.viewportCount, scissor);
  m_currentFrame.commandBuffer.setScissorWithCountEXT(m_currentFrame.viewportCount, scissors.data());
}

void RHIVK::bindComputePipeline(RHIComputePipeline::ptr) { RHI_VK_NOT_IMPLEMENTED(); }

// -------- Resource binding (push descriptors) --------

void RHIVK::resetTransientUniforms() {
  if (m_frameContextIndex < m_transientUniforms.size())
    m_transientUniforms[m_frameContextIndex].cursor = 0;
}

RHIVK::TransientBufferAlloc RHIVK::allocTransientUniform(const void* data, size_t size) {
  auto& ring = m_transientUniforms[m_frameContextIndex];
  // Align cursor up to UBO alignment.
  vk::DeviceSize offset = (ring.cursor + m_transientUniformAlignment - 1) & ~(m_transientUniformAlignment - 1);
  if (offset + size > ring.size) {
    fprintf(stderr, "RHIVK::allocTransientUniform: ring exhausted (need %zu @ %zu, size %zu)\n",
      size, size_t(offset), size_t(ring.size));
    abort();
  }
  memcpy(static_cast<uint8_t*>(ring.mapped) + offset, data, size);
  ring.cursor = offset + size;
  return TransientBufferAlloc{ring.buffer.get(), offset, size};
}

void RHIVK::loadUniformBlock(FxAtomicString name, RHIBuffer::ptr buffer) {
  if (!m_boundPipeline) {
    fprintf(stderr, "RHIVK::loadUniformBlock(\"%s\"): no pipeline bound\n", name.c_str());
    return;
  }
  const auto& bindings = m_boundPipeline->resourceBindings();
  auto it = bindings.find(name.c_str());
  if (it == bindings.end()) return; // shader does not use this block

  // Fan out the write to every binding slot this name occupies (typically
  // one, but a cross-stage-shared uniform block lands at multiple slots
  // because our per-stage binding-number offset puts each stage's
  // reference at a distinct slot — see RHIShaderVK::stageBindingOffset).
  vk::Buffer vkBuf = static_cast<RHIBufferVK*>(buffer.get())->vkBuffer();
  for (const auto& binding : it->second) {
    StagedDescriptorWrite w;
    w.set = binding.set;
    w.bufferInfo.buffer = vkBuf;
    w.bufferInfo.offset = 0;
    w.bufferInfo.range = buffer->size();
    w.write.dstBinding = binding.binding;
    w.write.dstArrayElement = 0;
    w.write.descriptorCount = 1;
    w.write.descriptorType = binding.descriptorType;
    m_pendingDescriptorWrites[{binding.set, binding.binding}] = w;
  }
  keepAliveForFrame(buffer);
}

void RHIVK::loadUniformBlockImmediate(FxAtomicString name, const void* data, size_t size) {
  if (!m_boundPipeline) return;
  const auto& bindings = m_boundPipeline->resourceBindings();
  auto it = bindings.find(name.c_str());
  if (it == bindings.end()) return;

  auto alloc = allocTransientUniform(data, size);
  for (const auto& binding : it->second) {
    StagedDescriptorWrite w;
    w.set = binding.set;
    w.bufferInfo.buffer = alloc.buffer;
    w.bufferInfo.offset = alloc.offset;
    w.bufferInfo.range = alloc.size;
    w.write.dstBinding = binding.binding;
    w.write.dstArrayElement = 0;
    w.write.descriptorCount = 1;
    w.write.descriptorType = binding.descriptorType;
    m_pendingDescriptorWrites[{binding.set, binding.binding}] = w;
  }
}

void RHIVK::loadShaderBuffer(FxAtomicString name, RHIBuffer::ptr buffer) {
  if (!m_boundPipeline) return;
  const auto& bindings = m_boundPipeline->resourceBindings();
  auto it = bindings.find(name.c_str());
  if (it == bindings.end()) return;

  vk::Buffer vkBuf = static_cast<RHIBufferVK*>(buffer.get())->vkBuffer();
  for (const auto& binding : it->second) {
    StagedDescriptorWrite w;
    w.set = binding.set;
    w.bufferInfo.buffer = vkBuf;
    w.bufferInfo.offset = 0;
    w.bufferInfo.range = buffer->size();
    w.write.dstBinding = binding.binding;
    w.write.dstArrayElement = 0;
    w.write.descriptorCount = 1;
    w.write.descriptorType = binding.descriptorType;
    m_pendingDescriptorWrites[{binding.set, binding.binding}] = w;
  }
  keepAliveForFrame(buffer);
}

void RHIVK::loadTexture(FxAtomicString name, RHISurface::ptr tex, RHISampler::ptr sampler) {
  if (!m_boundPipeline) return;
  if (!tex) {
    printf("RHIVK::loadTexture(\"%s\"): texture is null\n", name.c_str());
    return;
  }
  static const char* s_skipName = getenv("RHI_VK_SKIP_LOAD_TEXTURE");
  if (s_skipName && strcmp(s_skipName, name.c_str()) == 0) return;
  const auto& bindings = m_boundPipeline->resourceBindings();
  auto it = bindings.find(name.c_str());
  if (it == bindings.end()) return;

  vk::ImageView imageView = static_cast<RHISurfaceVK*>(tex.get())->vkImageView();
  // VK CIS descriptors require a real sampler handle. Callers that pass
  // sampler=nullptr (mirroring GL where the texture's own glTexParameter
  // state suffices) get a process-wide default linear/repeat sampler.
  // For bindings declared with an immutable sampler in the descriptor set
  // layout, the sampler half of the push is ignored by the spec — we still
  // pass the immutable handle defensively to keep validation/drivers happy.
  vk::Sampler defaultVkSampler = sampler
    ? static_cast<RHISamplerVK*>(sampler.get())->vkSampler()
    : defaultSampler();
  for (const auto& binding : it->second) {
    StagedDescriptorWrite w;
    w.set = binding.set;
    w.imageInfo.imageView = imageView;
    w.imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    w.imageInfo.sampler = binding.immutableSampler
      ? binding.immutableSampler->vkSampler()
      : defaultVkSampler;
    w.write.dstBinding = binding.binding;
    w.write.dstArrayElement = 0;
    w.write.descriptorCount = 1;
    w.write.descriptorType = binding.descriptorType;
    m_pendingDescriptorWrites[{binding.set, binding.binding}] = w;
  }
  keepAliveForFrame(tex);
  if (sampler) keepAliveForFrame(sampler);
}

vk::Sampler RHIVK::defaultSampler() {
  if (!m_defaultSampler) {
    vk::SamplerCreateInfo ci;
    ci.magFilter = vk::Filter::eLinear;
    ci.minFilter = vk::Filter::eLinear;
    ci.mipmapMode = vk::SamplerMipmapMode::eLinear;
    ci.addressModeU = vk::SamplerAddressMode::eRepeat;
    ci.addressModeV = vk::SamplerAddressMode::eRepeat;
    ci.addressModeW = vk::SamplerAddressMode::eRepeat;
    ci.maxLod = VK_LOD_CLAMP_NONE;
    m_defaultSampler = rhi()->vk()->device().createSamplerUnique(ci);
  }
  return m_defaultSampler.get();
}

void RHIVK::loadImage(FxAtomicString name, RHISurface::ptr tex, RHIImageAccessType, uint32_t /*mipLevel*/, int32_t /*layerIndex*/) {
  if (!m_boundPipeline) return;
  const auto& bindings = m_boundPipeline->resourceBindings();
  auto it = bindings.find(name.c_str());
  if (it == bindings.end()) return;

  vk::ImageView imageView = static_cast<RHISurfaceVK*>(tex.get())->vkImageView();
  for (const auto& binding : it->second) {
    StagedDescriptorWrite w;
    w.set = binding.set;
    w.imageInfo.imageView = imageView;
    w.imageInfo.imageLayout = vk::ImageLayout::eGeneral;
    w.write.dstBinding = binding.binding;
    w.write.dstArrayElement = 0;
    w.write.descriptorCount = 1;
    w.write.descriptorType = binding.descriptorType;
    m_pendingDescriptorWrites[{binding.set, binding.binding}] = w;
  }
  keepAliveForFrame(tex);
}

void RHIVK::flushDrawState() {
  vk::CommandBuffer cb = m_currentFrame.commandBuffer;
  if (!m_boundPipeline) {
    fprintf(stderr, "RHIVK::flushDrawState: no pipeline bound\n");
    abort();
  }

  // Bind stream buffers (one vkCmdBindVertexBuffers call per contiguous run
  // to keep the API surface clean; in practice we always bind from slot 0).
  if (!m_pendingStreamBuffers.empty()) {
    std::vector<vk::Buffer> buffers(m_pendingStreamBuffers.size());
    std::vector<vk::DeviceSize> offsets(m_pendingStreamBuffers.size());
    for (size_t i = 0; i < m_pendingStreamBuffers.size(); ++i) {
      buffers[i] = m_pendingStreamBuffers[i].buffer;
      offsets[i] = m_pendingStreamBuffers[i].offset;
    }
    cb.bindVertexBuffers(0, uint32_t(buffers.size()), buffers.data(), offsets.data());
  }

  // Group descriptor writes by set, then push each set with a single
  // vkCmdPushDescriptorSetKHR call. m_pendingDescriptorWrites is keyed
  // by (set, binding) so we already have one entry per slot — last
  // load*() to that slot wins.
  if (!m_pendingDescriptorWrites.empty()) {
    std::map<uint32_t, std::vector<vk::WriteDescriptorSet>> writesPerSet;
    for (auto& [key, sw] : m_pendingDescriptorWrites) {
      vk::WriteDescriptorSet w = sw.write;
      switch (sw.write.descriptorType) {
        case vk::DescriptorType::eSampledImage:
        case vk::DescriptorType::eStorageImage:
        case vk::DescriptorType::eSampler:
        case vk::DescriptorType::eCombinedImageSampler:
          w.pImageInfo = &sw.imageInfo;
          break;
        default:
          w.pBufferInfo = &sw.bufferInfo;
          break;
      }
      writesPerSet[sw.set].push_back(w);
    }
    for (const auto& [set, writes] : writesPerSet) {
      cb.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics,
        m_boundPipeline->pipelineLayout(), set,
        uint32_t(writes.size()), writes.data());
    }
  }
}

void RHIVK::drawPrimitives(uint32_t vertexStart, uint32_t vertexCount, uint32_t instanceCount, uint32_t baseInstance) {
  flushDrawState();
  if (instanceCount == 0) instanceCount = 1;
  static const bool s_trace = getenv("RHI_VK_TRACE_DRAWS") && atoi(getenv("RHI_VK_TRACE_DRAWS"));
  static const int s_skip = getenv("RHI_VK_SKIP_DRAWS") ? atoi(getenv("RHI_VK_SKIP_DRAWS")) : 0; // 1=skip all, 2=skip indexed only, 3=skip primitives only
  static uint32_t s_drawIndex = 0;
  if (s_trace) {
    fprintf(stderr, "[draw %u] drawPrimitives pipeline=%p vstart=%u vcount=%u inst=%u baseInst=%u streams=%zu descs=%zu\n",
      s_drawIndex++, static_cast<void*>(m_boundPipeline),
      vertexStart, vertexCount, instanceCount, baseInstance,
      m_pendingStreamBuffers.size(), m_pendingDescriptorWrites.size());
  }
  if (s_skip == 1 || s_skip == 3) return;
  m_currentFrame.commandBuffer.draw(vertexCount, instanceCount, vertexStart, baseInstance);
}
void RHIVK::drawPrimitivesIndirect(RHIBuffer::ptr indirectBuffer, uint32_t indirectCommandCount, uint32_t indirectCommandArrayOffset) {
  flushDrawState();
  RHIBufferVK* indirectBufVK = static_cast<RHIBufferVK*>(indirectBuffer.get());
  // VkDrawIndirectCommand is 16 bytes (vertexCount, instanceCount,
  // firstVertex, firstInstance). The engine uses the same struct layout
  // for GL and VK callers, so the offset is in array units of 16 bytes.
  constexpr uint32_t kStride = sizeof(VkDrawIndirectCommand);
  m_currentFrame.commandBuffer.drawIndirect(
    indirectBufVK->vkBuffer(),
    /*offset=*/ vk::DeviceSize(indirectCommandArrayOffset) * kStride,
    /*drawCount=*/ indirectCommandCount,
    /*stride=*/ kStride);
  keepAliveForFrame(indirectBuffer);
}
void RHIVK::drawIndexedPrimitives(RHIBuffer::ptr indexBuffer, RHIIndexBufferType indexBufferType,
  uint32_t indexCount, uint32_t indexOffsetElements, uint32_t instanceCount, uint32_t baseInstance) {
  flushDrawState();
  vk::IndexType type = (indexBufferType == kIndexBufferTypeUInt16) ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
  m_currentFrame.commandBuffer.bindIndexBuffer(
    static_cast<RHIBufferVK*>(indexBuffer.get())->vkBuffer(), 0, type);
  keepAliveForFrame(indexBuffer);
  if (instanceCount == 0) instanceCount = 1;
  static const bool s_trace = getenv("RHI_VK_TRACE_DRAWS") && atoi(getenv("RHI_VK_TRACE_DRAWS"));
  static const int s_skip = getenv("RHI_VK_SKIP_DRAWS") ? atoi(getenv("RHI_VK_SKIP_DRAWS")) : 0; // 1=skip all, 2=skip indexed only, 3=skip primitives only
  static const int s_maxIndexedDraws = getenv("RHI_VK_MAX_INDEXED_DRAWS") ? atoi(getenv("RHI_VK_MAX_INDEXED_DRAWS")) : -1;
  static uint32_t s_drawIndex = 0;
  uint32_t myIdx = s_drawIndex++;
  if (s_trace) {
    fprintf(stderr, "[draw %u] drawIndexed pipeline=%p icount=%u ioffset=%u inst=%u baseInst=%u streams=%zu descs=%zu\n",
      myIdx, static_cast<void*>(m_boundPipeline),
      indexCount, indexOffsetElements, instanceCount, baseInstance,
      m_pendingStreamBuffers.size(), m_pendingDescriptorWrites.size());
  }
  if (s_skip == 1 || s_skip == 2) return;
  if (s_maxIndexedDraws >= 0 && int(myIdx) >= s_maxIndexedDraws) return;
  m_currentFrame.commandBuffer.drawIndexed(indexCount, instanceCount, indexOffsetElements, 0, baseInstance);
}
void RHIVK::drawIndexedPrimitivesIndirect(RHIBuffer::ptr indexBuffer, RHIIndexBufferType indexBufferType,
  RHIBuffer::ptr indirectBuffer, uint32_t indirectCommandCount, uint32_t indirectCommandArrayOffset) {
  flushDrawState();
  vk::IndexType type = (indexBufferType == kIndexBufferTypeUInt16) ? vk::IndexType::eUint16 : vk::IndexType::eUint32;
  m_currentFrame.commandBuffer.bindIndexBuffer(
    static_cast<RHIBufferVK*>(indexBuffer.get())->vkBuffer(), 0, type);
  keepAliveForFrame(indexBuffer);

  RHIBufferVK* indirectBufVK = static_cast<RHIBufferVK*>(indirectBuffer.get());
  // VkDrawIndexedIndirectCommand is 20 bytes (indexCount, instanceCount,
  // firstIndex, vertexOffset, firstInstance) — matches GL's
  // DrawElementsIndirectCommand, so the engine's array offset (in command
  // units) translates 1:1.
  constexpr uint32_t kStride = sizeof(VkDrawIndexedIndirectCommand);
  m_currentFrame.commandBuffer.drawIndexedIndirect(
    indirectBufVK->vkBuffer(),
    /*offset=*/ vk::DeviceSize(indirectCommandArrayOffset) * kStride,
    /*drawCount=*/ indirectCommandCount,
    /*stride=*/ kStride);
  keepAliveForFrame(indirectBuffer);
}

RHITimerQuery::ptr RHIVK::newTimerQuery() {
  return RHITimerQuery::ptr(new RHITimerQueryVK());
}
RHIOcclusionQuery::ptr RHIVK::newOcclusionQuery(RHIOcclusionQueryMode mode) {
  return RHIOcclusionQuery::ptr(new RHIOcclusionQueryVK(mode));
}

void RHIVK::beginTimerQuery(RHITimerQuery::ptr query) {
  RHITimerQueryVK* q = static_cast<RHITimerQueryVK*>(query.get());
  // Host-reset both slots before re-writing this frame. Safe because the
  // engine reads getQueryResult (which blocks on GPU completion) before
  // re-beginning, so any prior writes are guaranteed complete here.
  rhi()->vk()->device().resetQueryPool(q->queryPool(), 0, 2);
  ensureFrameCBOpen();
  m_currentFrame.commandBuffer.writeTimestamp(
    vk::PipelineStageFlagBits::eTopOfPipe, q->queryPool(), 0);
  q->setKind(RHITimerQueryVK::Kind::kElapsedBegun);
  keepAliveForFrame(query);
}

void RHIVK::endTimerQuery(RHITimerQuery::ptr query) {
  RHITimerQueryVK* q = static_cast<RHITimerQueryVK*>(query.get());
  ensureFrameCBOpen();
  m_currentFrame.commandBuffer.writeTimestamp(
    vk::PipelineStageFlagBits::eBottomOfPipe, q->queryPool(), 1);
  q->setKind(RHITimerQueryVK::Kind::kElapsedEnded);
  keepAliveForFrame(query);
}

void RHIVK::recordTimestamp(RHITimerQuery::ptr query) {
  RHITimerQueryVK* q = static_cast<RHITimerQueryVK*>(query.get());
  rhi()->vk()->device().resetQueryPool(q->queryPool(), 0, 2);
  ensureFrameCBOpen();
  m_currentFrame.commandBuffer.writeTimestamp(
    vk::PipelineStageFlagBits::eBottomOfPipe, q->queryPool(), 0);
  q->setKind(RHITimerQueryVK::Kind::kTimestamp);
  keepAliveForFrame(query);
}

void RHIVK::beginOcclusionQuery(RHIOcclusionQuery::ptr query) {
  if (!m_currentFrame.passActive) return;
  RHIOcclusionQueryVK* q = static_cast<RHIOcclusionQueryVK*>(query.get());
  rhi()->vk()->device().resetQueryPool(q->queryPool(), 0, 1);
  // Both query modes map to plain occlusion: kAnySamplesPassed corresponds
  // to non-PRECISE, kSampleCount corresponds to PRECISE (the result is the
  // actual sample count rather than 0/1). The flag is on beginQuery.
  vk::QueryControlFlags flags = (q->mode() == kOcclusionQueryModeSampleCount)
    ? vk::QueryControlFlagBits::ePrecise
    : vk::QueryControlFlags();
  m_currentFrame.commandBuffer.beginQuery(q->queryPool(), 0, flags);
  keepAliveForFrame(query);
}

void RHIVK::endOcclusionQuery(RHIOcclusionQuery::ptr query) {
  if (!m_currentFrame.passActive) return;
  RHIOcclusionQueryVK* q = static_cast<RHIOcclusionQueryVK*>(query.get());
  m_currentFrame.commandBuffer.endQuery(q->queryPool(), 0);
  keepAliveForFrame(query);
}

uint64_t RHIVK::getQueryResult(RHIOcclusionQuery::ptr query) {
  RHIOcclusionQueryVK* q = static_cast<RHIOcclusionQueryVK*>(query.get());
  uint64_t value = 0;
  auto result = rhi()->vk()->device().getQueryPoolResults(
    q->queryPool(), 0, 1,
    sizeof(uint64_t), &value, sizeof(uint64_t),
    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
  if (result != vk::Result::eSuccess) {
    // eNotReady can't happen with eWait, but be defensive.
    return 0;
  }
  return value;
}

uint64_t RHIVK::getQueryResult(RHITimerQuery::ptr query) {
  RHITimerQueryVK* q = static_cast<RHITimerQueryVK*>(query.get());
  if (q->kind() == RHITimerQueryVK::Kind::kEmpty) {
    // Mirrors GL behavior: querying a never-written timer reads as 0
    // (engine treats this as "no sample yet" on the first frame).
    return 0;
  }
  // If the frame CB containing the writeTimestamp(s) hasn't been
  // submitted yet, the eWait below would hang forever. Engine pattern is
  // "read frame N-1's result at the top of frame N, after swapBuffers
  // submitted N-1." Catch the misuse loudly instead.
  if (m_currentFrame.cbOpen) {
    fprintf(stderr, "RHIVK::getQueryResult: called while frame CB is still open — "
                    "timestamp writes have not been submitted, so eWait would hang. "
                    "Call after swapBuffers() (or flush()).\n");
    abort();
  }
  uint32_t slotCount = (q->kind() == RHITimerQueryVK::Kind::kTimestamp) ? 1 : 2;
  uint64_t values[2] = {0, 0};
  auto result = rhi()->vk()->device().getQueryPoolResults(
    q->queryPool(), 0, slotCount,
    sizeof(uint64_t) * slotCount, values, sizeof(uint64_t),
    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
  if (result != vk::Result::eSuccess) {
    return 0;
  }
  q->setKind(RHITimerQueryVK::Kind::kEmpty);
  if (slotCount == 1) {
    // recordTimestamp: return the absolute timestamp.
    // (Apply timestampPeriod to convert ticks → nanoseconds, even though
    // it's 1 on this hardware. Stays correct if cached for portability.)
    return uint64_t(double(values[0]) * m_timestampPeriodNs);
  }
  // begin/endTimerQuery: return elapsed nanoseconds.
  uint64_t deltaTicks = values[1] - values[0];
  return uint64_t(double(deltaTicks) * m_timestampPeriodNs);
}

uint64_t RHIVK::getTimestampImmediate() {
  // Synchronous "what's the GPU clock now" — used by the engine for ad-hoc
  // timing. Allocate a one-shot pool + CB, write timestamp, submit, wait,
  // read out. Heavyweight on purpose: callers should prefer
  // begin/endTimerQuery for hot-path timing.
  vk::Device device = rhi()->vk()->device();
  vk::QueryPoolCreateInfo qci{vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, 1};
  vk::UniqueQueryPool pool = device.createQueryPoolUnique(qci);
  device.resetQueryPool(pool.get(), 0, 1);

  vk::CommandBufferAllocateInfo cbAi{m_commandPool.get(), vk::CommandBufferLevel::ePrimary, 1};
  auto cbs = device.allocateCommandBuffersUnique(cbAi);
  vk::CommandBuffer cb = cbs[0].get();
  cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  cb.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, pool.get(), 0);
  cb.end();

  vk::SubmitInfo submit;
  submit.setCommandBuffers(cb);
  rhi()->vk()->queue().submit(submit, vk::Fence());
  rhi()->vk()->queue().waitIdle();

  uint64_t value = 0;
  (void) device.getQueryPoolResults(pool.get(), 0, 1,
    sizeof(uint64_t), &value, sizeof(uint64_t),
    vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
  return uint64_t(double(value) * m_timestampPeriodNs);
}

bool RHIVK::getTimerQueryDisjointState() {
  // GL_GPU_DISJOINT_EXT signals that ES timestamps may be inaccurate due
  // to GPU power-state changes. Vulkan exposes timestampValidBits (all 64
  // on this hardware) and doesn't surface a comparable transient signal —
  // callers treat "false" as "trust the measurement."
  return false;
}

RHIFence::ptr RHIVK::registerFrameCompletionFence() {
  // Make sure a frame CB is open so the registration ties to a real frame that
  // a subsequent swapBuffers/flush will submit. In practice this is called
  // mid-frame (after the caller's render passes) so the CB is already open and
  // this is a no-op.
  ensureFrameCBOpen();
  // The next swapBuffers/flush submit signals m_frameCompletionValue + 1 on the
  // frame-completion timeline (see swapBuffers/flush). Hand the consumer a fence
  // that host-waits on exactly that value. Multiple registrations in the same
  // frame share the value -- they all resolve at that frame's submit.
  return RHIFence::ptr(new RHIFenceVK(m_frameCompletionTimeline.get(), m_frameCompletionValue + 1));
}

void RHIVK::dispatchCompute(uint32_t, uint32_t, uint32_t) { RHI_VK_NOT_IMPLEMENTED(); }
void RHIVK::dispatchComputeIndirect(RHIBuffer::ptr) { RHI_VK_NOT_IMPLEMENTED(); }

void RHIVK::pushDebugGroup(const char*) { RHI_VK_NOT_IMPLEMENTED(); }
void RHIVK::popDebugGroup() { RHI_VK_NOT_IMPLEMENTED(); }

void RHIVK::swapBuffers(RHIRenderTarget::ptr /*target*/) {
  if (m_currentFrame.passActive) {
    fprintf(stderr, "RHIVK::swapBuffers: render pass still open — call endRenderPass first\n");
    abort();
  }
  if (!m_currentFrame.cbOpen) {
    // No begin* was called this frame. Submitting an empty CB would be
    // harmless but the swap chain still expects a present that consumes
    // the acquired image; we never acquired one, so there's nothing to
    // present either. Engine convention: a clear-only frame is expressed
    // as beginRenderPass(target, kLoadClear) + endRenderPass(target).
    fprintf(stderr, "RHIVK::swapBuffers: no command buffer recorded this frame "
                    "(if you wanted to render an empty frame, call beginRenderPass(target, kLoadClear) + endRenderPass(target))\n");
    abort();
  }
  if (!m_currentFrame.hasWindowImage) {
    fprintf(stderr, "RHIVK::swapBuffers: no window-RT pass this frame — nothing to present\n");
    abort();
  }

  vk::CommandBuffer cb = m_currentFrame.commandBuffer;
  cb.end();

  // Submit: wait on imageAcquired (colour attachment output stage),
  // signal renderFinished (for present) + per-frame fence (for CB recycling).

  // For interop sync, also wait on the interop timeline at
  // CUDA's latest signaled value, and signal the next VK value on it. The
  // wait stage on the interop semaphore is eAllCommands — interop surfaces
  // can be sampled in any stage, so we conservatively pessimize until
  // profiling shows it matters.
  // Signals: renderFinished (binary, present) + interop timeline + frame-
  // completion timeline. Waits: imageAcquired (binary, only if the frame source
  // provides one) + interop timeline.
  std::array<vk::Semaphore, 2> waitSemaphores{vk::Semaphore{}, vk::Semaphore{}};
  std::array<vk::PipelineStageFlags, 2> waitStages{
    vk::PipelineStageFlagBits::eColorAttachmentOutput,
    vk::PipelineStageFlagBits::eAllCommands};
  std::array<uint64_t, 2> waitValues{0, 0};
  std::array<vk::Semaphore, 3> signalSemaphores{m_currentFrame.renderFinished, vk::Semaphore{}, vk::Semaphore{}};
  std::array<uint64_t, 3> signalValues{0, 0, 0};
  uint32_t waitCount = 0;
  uint32_t signalCount = 1;

  // The frame source may host-synchronize the acquire (fence-based) and hand
  // back a null imageAcquired — in which case there is no GPU wait to add. When
  // it is present, wait on it at the colour-attachment-output stage.
  if (m_currentFrame.imageAcquired) {
    waitSemaphores[waitCount] = m_currentFrame.imageAcquired;
    waitStages[waitCount] = vk::PipelineStageFlagBits::eColorAttachmentOutput;
    ++waitCount;
  }

  // Interop sync: wait on the CUDA->VK timeline at CUDA's latest signaled
  // value. The timeline is CUDA-signal-only -- VK does NOT signal it. Nothing
  // waits on a VK-signaled interop value (signalRHIToCUDA is unused), so a VK
  // signal here would only make the timeline a fragile two-writer object: CUDA
  // and VK interleaving values on one timeline can wedge it (the producer/
  // consumer lockstep breaks at shutdown and the timeline stalls below the
  // last CUDA value, stranding this very wait). Keeping it single-writer lets
  // the host force-complete it cleanly at teardown (see waitForGPUIdle).
  {
    waitStages[waitCount] = vk::PipelineStageFlagBits::eAllCommands;
    waitSemaphores[waitCount] = m_interopTimeline.semaphore.get();
    waitValues[waitCount] = m_interopTimelineCudaSignaledValue;
    ++waitCount;
  }

  // Signal the frame-completion timeline so any RHIFence registered this frame
  // resolves on this submit's GPU completion.
  signalSemaphores[signalCount] = m_frameCompletionTimeline.get();
  signalValues[signalCount] = ++m_frameCompletionValue;
  ++signalCount;
  // Signal the VK->CUDA interop timeline so a later signalRHIToCUDA() can make a
  // CUDA stream wait on this frame's GPU completion. Single-writer (VK only).
  signalSemaphores[signalCount] = m_rhiToCudaTimeline.semaphore.get();
  signalValues[signalCount] = rhiToCudaAllocateSignalValue();
  ++signalCount;
  // Timeline submit info: pValues entries are ignored for binary semaphores
  // (imageAcquired / renderFinished), but counts must match the outer
  // VkSubmitInfo's wait / signal counts when the chain is present.
  vk::TimelineSemaphoreSubmitInfo timelineSubmit{};
  timelineSubmit.setWaitSemaphoreValues(vk::ArrayProxyNoTemporaries<const uint64_t>(waitCount, waitValues.data()));
  timelineSubmit.setSignalSemaphoreValues(vk::ArrayProxyNoTemporaries<const uint64_t>(signalCount, signalValues.data()));
  vk::SubmitInfo submit;
  submit.setWaitSemaphores(vk::ArrayProxyNoTemporaries<const vk::Semaphore>(waitCount, waitSemaphores.data()));
  submit.setWaitDstStageMask(vk::ArrayProxyNoTemporaries<const vk::PipelineStageFlags>(waitCount, waitStages.data()));
  submit.setCommandBuffers(cb);
  submit.setSignalSemaphores(vk::ArrayProxyNoTemporaries<const vk::Semaphore>(signalCount, signalSemaphores.data()));
  submit.setPNext(&timelineSubmit);
  rhi()->vk()->queue().submit(submit, m_currentFrame.frameFence);

  // Present via the frame source. Backend manages its own per-frame index.
  VKFrameInfo info{};
  info.swapchainIndex = m_currentFrame.swapchainIndex;
  info.swapchainImage = m_currentFrame.swapImage;
  info.extent = m_currentFrame.swapExtent;
  info.format = m_currentFrame.swapFormat;
  info.imageAcquired = m_currentFrame.imageAcquired;
  info.renderFinished = m_currentFrame.renderFinished;
  m_frameSource->presentVKFrame(info);

  // Drain debug PNG dumps before destroying their staging buffers (the
  // CurrentFrame reset below releases the UniqueBuffer/UniqueDeviceMemory).
  processPendingDumps();

  // Advance the slot ring.
  m_frameContextIndex = (m_frameContextIndex + 1) % m_frameContexts.size();
  m_currentFrame = {}; // cbOpen=false, hasWindowImage=false, passActive=false
  m_boundPipeline = nullptr;
  m_pendingStreamBuffers.clear();
  m_pendingDescriptorWrites.clear();
  ++m_dumpFrameCounter;
  m_passCounterThisFrame = 0;
}
void RHIVK::flush() {
  // Submit the currently-recording frame CB without presenting. Used when
  // a caller wants the GPU to start executing work before swapBuffers
  // (rare — the engine doesn't call this on the hot path). Advances the
  // slot ring so subsequent begin* picks the next slot.
  if (m_currentFrame.passActive) {
    fprintf(stderr, "RHIVK::flush: render pass still open — call endRenderPass first\n");
    abort();
  }
  if (m_currentFrame.hasWindowImage) {
    fprintf(stderr, "RHIVK::flush: window-RT pass in flight — use swapBuffers, not flush\n");
    abort();
  }
  if (!m_currentFrame.cbOpen) {
    // Nothing to do.
    return;
  }
  vk::CommandBuffer cb = m_currentFrame.commandBuffer;
  cb.end();
  // Same interop-timeline plumbing as swapBuffers, minus the per-swap-image
  // imageAcquired/renderFinished pair (flush() is for off-screen frames with no
  // present). Wait on the CUDA->VK interop timeline (CUDA-signal-only; VK does
  // not signal it -- see swapBuffers). Signal the frame-completion timeline and
  // the VK->CUDA interop timeline (both VK-signal-only).
  vk::Semaphore interopSem = m_interopTimeline.semaphore.get();
  vk::PipelineStageFlags interopWaitStage = vk::PipelineStageFlagBits::eAllCommands;
  uint64_t interopWaitValue = m_interopTimelineCudaSignaledValue;

  std::array<vk::Semaphore, 2> signalSemaphores{m_frameCompletionTimeline.get(), m_rhiToCudaTimeline.semaphore.get()};
  std::array<uint64_t, 2> signalValues{++m_frameCompletionValue, rhiToCudaAllocateSignalValue()};

  vk::TimelineSemaphoreSubmitInfo timelineSubmit{};
  timelineSubmit.setWaitSemaphoreValues(interopWaitValue);
  timelineSubmit.setSignalSemaphoreValues(vk::ArrayProxyNoTemporaries<const uint64_t>(2, signalValues.data()));

  vk::SubmitInfo submit;
  submit.setCommandBuffers(cb);
  submit.setWaitSemaphores(interopSem);
  submit.setWaitDstStageMask(interopWaitStage);
  submit.setSignalSemaphores(vk::ArrayProxyNoTemporaries<const vk::Semaphore>(2, signalSemaphores.data()));
  submit.setPNext(&timelineSubmit);

  rhi()->vk()->queue().submit(submit, m_currentFrame.frameFence);

  processPendingDumps();

  m_frameContextIndex = (m_frameContextIndex + 1) % m_frameContexts.size();
  m_currentFrame = {};
  m_boundPipeline = nullptr;
  m_pendingStreamBuffers.clear();
  m_pendingDescriptorWrites.clear();
  ++m_dumpFrameCounter;
  m_passCounterThisFrame = 0;
}
void RHIVK::flushAndWaitForGPUScheduling() { RHI_VK_NOT_IMPLEMENTED(); }

void RHIVK::waitForGPUIdle() {
  // Force-complete both interop timelines from the host so neither direction can
  // deadlock the drain. Both are single-writer (see header), so jumping each
  // forward to its highest allocated value is safe -- there is no producer
  // signal to invalidate, and shutdown discards in-flight frame data anyway. At
  // shutdown a timeline can stall below its last allocated value (a producer's
  // final async signals don't land once the render loop stops feeding the
  // pipeline), stranding the consumer on its wait -> GPU host watchdog -> device
  // lost; force-signaling forward releases it. vkSignalSemaphore requires a
  // strictly-greater value, so this only signals when actually behind.
  auto forceTimelineForward = [](RHIVulkan::ExternalSemaphore& sem, uint64_t target, const char* label) {
    if (!sem.semaphore)
      return;
    try {
      vk::Device dev = rhi()->vk()->device();
      uint64_t cur = dev.getSemaphoreCounterValue(sem.semaphore.get());
      if (cur < target) {
        vk::SemaphoreSignalInfo si{};
        si.semaphore = sem.semaphore.get();
        si.value = target;
        dev.signalSemaphore(si);
        fprintf(stderr, "RHIVK::waitForGPUIdle: host-signaled %s interop timeline %llu -> %llu to unblock stranded waits\n",
          label, (unsigned long long) cur, (unsigned long long) target);
      }
    } catch (const vk::SystemError& ex) {
      fprintf(stderr, "RHIVK::waitForGPUIdle: %s interop timeline host-signal failed (%s)\n", label, ex.what());
    }
  };

  if (supportsCUDAInterop()) {
    // VK->CUDA first, so the cuCtxSynchronize() below can't block on a CUDA
    // stream still waiting on a VK value.
    forceTimelineForward(m_rhiToCudaTimeline, m_rhiToCudaNextValue, "VK->CUDA");
    // Drain CUDA: lets CUDA's pending CUDA->VK signals land before we release
    // the VK side. Result dropped on purpose -- on a clean shutdown it succeeds,
    // and once the device is lost there's nothing to recover.
    (void) cuCtxSynchronize();
    // CUDA->VK: release any VK submit still parked on a CUDA value.
    forceTimelineForward(m_interopTimeline, m_interopTimelineNextValue, "CUDA->VK");
  }

  // Now drain the in-flight per-frame fences with a single bounded timeout
  // (well under the GPU's ~2 s acquire watchdog) rather than device.waitIdle().
  // Frames that can complete drain promptly; anything still parked times out and
  // we return so teardown proceeds before the watchdog can lose the device.
  // Fences for never-used slots were created signaled, so they don't extend it.
  vk::Device device = rhi()->vk()->device();
  std::vector<vk::Fence> fences;
  fences.reserve(m_frameContexts.size());
  for (auto& fc : m_frameContexts) {
    if (fc.frameFence)
      fences.push_back(fc.frameFence.get());
  }
  if (fences.empty())
    return;

  constexpr uint64_t kDrainTimeoutNs = 200'000'000ull; // 200 ms << ~2 s GPU acquire watchdog
  try {
    vk::Result r = device.waitForFences(fences, /*waitAll=*/ VK_TRUE, kDrainTimeoutNs);
    if (r == vk::Result::eTimeout) {
      fprintf(stderr, "RHIVK::waitForGPUIdle: %zu frame fence(s) still unsignaled after %llu ms -- "
                      "a submit is likely parked on the swapchain imageAcquired semaphore. "
                      "Proceeding to teardown before the GPU host watchdog fires.\n",
        fences.size(), (unsigned long long) (kDrainTimeoutNs / 1'000'000ull));
    }
  } catch (const vk::SystemError& ex) {
    fprintf(stderr, "RHIVK::waitForGPUIdle: waitForFences failed (%s); device may already be lost.\n", ex.what());
    m_deviceLost = true;
  }
}
uint32_t RHIVK::maxMultisampleSamples() { RHI_VK_NOT_IMPLEMENTED(); }
bool RHIVK::supportsGeometryShaders() { RHI_VK_NOT_IMPLEMENTED(); }

// ---------- Interop support ----------
bool RHIVK::supportsCUDAInterop() {
  return true;
}

void RHIVK::signalCUDAToRHI(CUstream stream) {
  // CUDA-side signal on the CUDA->VK timeline. Allocate the next value from this
  // timeline's own counter (CUDA is its only writer). The RHI consumer
  // (swapBuffers / flush) reads m_interopTimelineCudaSignaledValue as its wait
  // value when it builds the next submit's TimelineSemaphoreSubmitInfo.
  m_interopTimelineCudaSignaledValue = ++m_interopTimelineNextValue;
  cudaExternalSemaphoreSignalParams params = {};
  params.params.fence.value = m_interopTimelineCudaSignaledValue;
  CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&m_interopTimelineCU, &params, 1, stream));
}

void RHIVK::signalRHIToCUDA(CUstream stream) {
  // CUDA-side wait on the VK->CUDA timeline: make `stream` block until VK has
  // finished the work submitted up to the most recent swapBuffers/flush. VK
  // signals that timeline at every such submit (see swapBuffers / flush), so we
  // wait on its latest signaled value. If no VK submit has happened yet,
  // m_rhiToCudaSignaledValue == 0 and the wait resolves immediately ("nothing
  // to wait on").
  //
  // Will-not-hang-at-shutdown: this timeline is VK-signal-only, so waitForGPUIdle()
  // can host-signal it forward to release any still-pending CUDA wait before it
  // drains CUDA -- the symmetric escape hatch to the CUDA->VK direction. As long
  // as either VK reaches the value or the host force-signals it, cuCtxSynchronize
  // can never deadlock on this wait.
  cudaExternalSemaphoreWaitParams params = {};
  params.params.fence.value = m_rhiToCudaSignaledValue;
  CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&m_rhiToCudaTimelineCU, &params, 1, stream));
}

RHIBuffer::ptr RHIVK::newInteropBuffer(size_t size, RHIBufferUsageMode mode) {
  return RHIBuffer::ptr(RHIInteropBufferVK::newBuffer(size, mode));
}
RHISurface::ptr RHIVK::newInteropSurface(uint32_t w, uint32_t h, const RHISurfaceDescriptor& d) {
  return RHISurface::ptr(RHIInteropSurfaceVK::newTexture2D(w, h, d));
}


// ----------- Misc internals ----------

RHIShader::ptr RHIVK::internalCompileShader(const RHIShaderDescriptor& descriptor) {
  return RHIShaderVK::compileFromDescriptor(descriptor);
}
RHIRenderPipeline::ptr RHIVK::internalCompileRenderPipeline(RHIShader::ptr shader, const RHIRenderPipelineDescriptor& d) {
  // Phase 4: defer VkShaderEXT + descriptor-layout creation until first bind.
  // initRHIResources compiles many pipelines; not all will be drawn.
  RHIShaderVK::ptr shaderVK(static_cast<RHIShaderVK*>(shader.get()));
  return RHIRenderPipeline::ptr(new RHIRenderPipelineVK(shaderVK, d));
}
RHIComputePipeline::ptr RHIVK::internalCompileComputePipeline(RHIShader::ptr) { RHI_VK_NOT_IMPLEMENTED(); }

// -------- Debug: per-frame render-target PNG dump --------

namespace {

// Returns (bytesPerPixel, stbComponents, formatTag). bpp==0 means unsupported.
struct DumpFormatInfo {
  uint32_t bpp = 0;
  int stbComponents = 0;
  const char* tag = "unknown";
  bool srcIsBGR = false; // swap channels[0]/[2] on write
  bool downconvert16To8 = false; // shift each u16 channel right 8
};

DumpFormatInfo classifyDumpFormat(vk::Format f) {
  switch (f) {
    case vk::Format::eB8G8R8A8Srgb: return {4, 4, "bgra8_srgb", true, false};
    case vk::Format::eB8G8R8A8Unorm: return {4, 4, "bgra8_unorm", true, false};
    case vk::Format::eR8G8B8A8Srgb: return {4, 4, "rgba8_srgb", false, false};
    case vk::Format::eR8G8B8A8Unorm: return {4, 4, "rgba8_unorm", false, false};
    // Packed A8B8G8R8: on little-endian (which is every platform we care
    // about) bytes 0..3 are R, G, B, A — identical to R8G8B8A8 in memory.
    case vk::Format::eA8B8G8R8UnormPack32: return {4, 4, "abgr8_unorm_pack32", false, false};
    case vk::Format::eA8B8G8R8SrgbPack32: return {4, 4, "abgr8_srgb_pack32", false, false};
    case vk::Format::eR8Unorm: return {1, 1, "r8_unorm", false, false};
    case vk::Format::eR8G8Unorm: return {2, 2, "rg8_unorm", false, false};
    case vk::Format::eR16Unorm: return {2, 1, "r16_unorm", false, true};
    case vk::Format::eR16G16Unorm: return {4, 2, "rg16_unorm", false, true};
    case vk::Format::eR16G16Sfloat: return {4, 2, "rg16_sfloat", false, true}; // crude: shifts mantissa+exp bits
    default: return {};
  }
}

} // namespace

void RHIVK::enqueuePassDump() {
  DumpFormatInfo fmt = classifyDumpFormat(m_currentFrame.passFormat);
  if (fmt.bpp == 0) {
    fprintf(stderr, "RHIVK: pass %u dump skipped: unsupported VkFormat (%u)\n",
      m_passCounterThisFrame, static_cast<uint32_t>(m_currentFrame.passFormat));
    return;
  }
  const uint32_t w = m_currentFrame.passExtent.width;
  const uint32_t h = m_currentFrame.passExtent.height;
  const vk::DeviceSize size = vk::DeviceSize(w) * h * fmt.bpp;

  vk::Device device = rhi()->vk()->device();
  vk::BufferCreateInfo bci{vk::BufferCreateFlags(),
    size, vk::BufferUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive};
  CurrentFrame::PendingDump pd;
  pd.buffer = device.createBufferUnique(bci);
  vk::MemoryRequirements memReq = device.getBufferMemoryRequirements(pd.buffer.get());
  vk::MemoryAllocateInfo mai{memReq.size,
    rhi()->vk()->findMemoryType(memReq.memoryTypeBits,
      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)};
  pd.memory = device.allocateMemoryUnique(mai);
  device.bindBufferMemory(pd.buffer.get(), pd.memory.get(), 0);
  pd.size = size;
  pd.width = w;
  pd.height = h;
  pd.format = m_currentFrame.passFormat;
  char filename[256];
  snprintf(filename, sizeof(filename),
    "%s/frame_%06llu_pass_%03u_%s_%ux%u_%s.png",
    m_dumpDir.c_str(),
    (unsigned long long) m_dumpFrameCounter,
    m_passCounterThisFrame,
    m_currentFrame.passIsWindowRT ? "window" : "offscreen",
    w, h, fmt.tag);
  pd.filepath = filename;

  // Transition passImageLayout -> TRANSFER_SRC_OPTIMAL, copy, transition back.
  // Source-stage of the toFinal barrier was eColorAttachmentOutput; we
  // need the writes to be visible to TRANSFER reads, so srcStage matches
  // the toFinal transition's dstStage.
  vk::CommandBuffer cb = m_currentFrame.commandBuffer;
  vk::ImageMemoryBarrier toSrc{
    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
    vk::AccessFlagBits::eTransferRead,
    m_currentFrame.passImageLayout,
    vk::ImageLayout::eTransferSrcOptimal,
    VK_QUEUE_FAMILY_IGNORED,
    VK_QUEUE_FAMILY_IGNORED,
    m_currentFrame.passImage,
    vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };
  cb.pipelineBarrier(
    vk::PipelineStageFlagBits::eAllCommands,
    vk::PipelineStageFlagBits::eTransfer,
    vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &toSrc);

  vk::BufferImageCopy region{
  /*bufferOffset=*/ 0,
 /*bufferRowLength=*/ 0, // tightly packed
  /*bufferImageHeight=*/ 0,
    vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, 0, 0, 1},
    vk::Offset3D{                              0, 0, 0  },
    vk::Extent3D{                              w, h, 1  },
  };
  cb.copyImageToBuffer(m_currentFrame.passImage,
    vk::ImageLayout::eTransferSrcOptimal,
    pd.buffer.get(), 1, &region);

  vk::ImageMemoryBarrier toBack{
    vk::AccessFlagBits::eTransferRead,
    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
    vk::ImageLayout::eTransferSrcOptimal,
    m_currentFrame.passImageLayout,
    VK_QUEUE_FAMILY_IGNORED,
    VK_QUEUE_FAMILY_IGNORED,
    m_currentFrame.passImage,
    vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };
  cb.pipelineBarrier(
    vk::PipelineStageFlagBits::eTransfer,
    vk::PipelineStageFlagBits::eAllCommands,
    vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &toBack);

  m_currentFrame.pendingDumps.push_back(std::move(pd));
}

void RHIVK::processPendingDumps() {
  if (m_currentFrame.pendingDumps.empty()) return;

  vk::Device device = rhi()->vk()->device();

  // Allocate a dedicated fence for this dump batch and queue an empty
  // signal-only submit. The worker waits on this fence rather than the
  // per-slot frame fence — the main thread will reset that one on its
  // next wrap around the ring, racing the worker's wait. The empty
  // submit signals after the just-submitted frame CB completes (queue
  // submissions execute in submission order), so by the time this fence
  // signals, every dump's staging buffer is fully written.
  vk::UniqueFence dumpFence = device.createFenceUnique({});
  rhi()->vk()->queue().submit({}, dumpFence.get());

  // Lazy worker startup: off-frames pay nothing, no thread spawns unless
  // and until the first dump batch arrives.
  if (!m_dumpWorker.joinable()) {
    m_dumpWorker = std::thread(&RHIVK::dumpWorkerThread, this);
  }

  {
    std::lock_guard<std::mutex> lock(m_dumpMutex);
    m_dumpQueue.push_back(DumpJob{std::move(dumpFence), std::move(m_currentFrame.pendingDumps)});
  }
  m_dumpCv.notify_one();
}

void RHIVK::dumpWorkerThread() {
  vk::Device device = rhi()->vk()->device();

  while (true) {
    DumpJob job;
    {
      std::unique_lock<std::mutex> lock(m_dumpMutex);
      m_dumpCv.wait(lock, [&] { return m_dumpStop || !m_dumpQueue.empty(); });
      // Drain any remaining jobs even after stop is signaled — staging
      // buffers / fences still need the device to be valid when their
      // Unique handles destruct.
      if (m_dumpQueue.empty()) return;
      job = std::move(m_dumpQueue.front());
      m_dumpQueue.pop_front();
    }

    vk::Fence fenceHandle = job.fence.get();
    (void) device.waitForFences(1, &fenceHandle, VK_TRUE, UINT64_MAX);

    for (auto& dump : job.dumps) {
      DumpFormatInfo fmt = classifyDumpFormat(dump.format);
      if (fmt.bpp == 0) continue; // shouldn't happen — checked at enqueue

      void* mapped = device.mapMemory(dump.memory.get(), 0, dump.size);

      // Convert into a tightly-packed RGB(A)8 buffer for stbi_write_png.
      std::vector<uint8_t> out(size_t(dump.width) * dump.height * fmt.stbComponents);
      const size_t pixels = size_t(dump.width) * dump.height;

      if (fmt.downconvert16To8) {
        // u16 → u8 by taking the high byte. For UNORM this is mathematically
        // correct (high byte = floor(value * 255 / 65535) within rounding).
        // For SFLOAT it's nonsense but at least produces a recognisable
        // image — half-float bit-pattern's exponent + top mantissa bits.
        const uint16_t* src = static_cast<const uint16_t*>(mapped);
        for (size_t i = 0; i < pixels * fmt.stbComponents; ++i) {
          out[i] = uint8_t(src[i] >> 8);
        }
      } else if (fmt.srcIsBGR) {
        // BGRA8 → RGBA8.
        const uint8_t* src = static_cast<const uint8_t*>(mapped);
        for (size_t i = 0; i < pixels; ++i) {
          out[i * 4 + 0] = src[i * 4 + 2];
          out[i * 4 + 1] = src[i * 4 + 1];
          out[i * 4 + 2] = src[i * 4 + 0];
          out[i * 4 + 3] = src[i * 4 + 3];
        }
      } else {
        memcpy(out.data(), mapped, pixels * fmt.bpp);
      }

      device.unmapMemory(dump.memory.get());

      stbi_write_png_compression_level = 3; // Default to faster PNG compression
      int ok = stbi_write_png(dump.filepath.c_str(),
        int(dump.width), int(dump.height), fmt.stbComponents,
        out.data(), int(dump.width * fmt.stbComponents));
      if (!ok) {
        fprintf(stderr, "RHIVK: stbi_write_png failed for %s\n", dump.filepath.c_str());
      } else {
        printf("RHIVK: dumped %s\n", dump.filepath.c_str());
      }
    }
    // job's UniqueFence + PendingDump UniqueBuffers/UniqueDeviceMemory
    // destruct here, releasing GPU resources back to the driver.
  }
}

// -------- Top-level VK backend init --------

void initRHIVulkan(const std::vector<const char*>& extraInstanceExtensions) {
  // No GL context to UUID-match against; select the first enumerated device.
  std::array<uint8_t, VK_UUID_SIZE> uuid{};
  if (!rhiVulkanInstallSingleton(uuid, extraInstanceExtensions)) {
    fprintf(stderr, "initRHIVulkan: RHIVulkan::create() failed; cannot bring up VK RHI backend\n");
    abort();
  }
  initRHI(new RHIVK());
}
