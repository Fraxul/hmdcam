#include "rhi/vk/RHIVK.h"
#include "rhi/vk/RHIBlendStateVK.h"
#include "rhi/vk/RHIBufferVK.h"
#include "rhi/vk/RHIDepthStencilStateVK.h"
#include "rhi/vk/RHIInteropBufferVK.h"
#include "rhi/vk/RHIInteropSyncDescriptor.h"
#include "rhi/vk/RHIQueryVK.h"
#include "rhi/vk/RHIRenderPipelineVK.h"
#include "rhi/vk/RHIRenderTargetVK.h"
#include "rhi/vk/RHISamplerVK.h"
#include "rhi/vk/RHIShaderVK.h"
#include "rhi/vk/RHISurfaceVK.h"
#include "rhi/vk/RHIVKFrameSource.h"
#include "rhi/vk/RHIVulkan.h"
#include "rhi/vk/RHIWindowRenderTargetVK.h"
#include <assert.h>
#include <cstring>
#include <stdio.h>
#include <stdlib.h>

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
  // Wait for any in-flight frames before tearing down command pool / image
  // views so the destruction order is safe.
  if (rhi() && rhi()->vk()) {
    rhi()->vk()->device().waitIdle();
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
RHIBuffer::ptr RHIVK::newInteropBuffer(size_t size, RHIBufferUsageMode mode, const RHIInteropSyncDescriptor& sync) {
  return RHIBuffer::ptr(RHIInteropBufferVK::newBuffer(size, mode, sync));
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

void RHIVK::beginRenderPass(RHIRenderTarget::ptr target,
  RHIRenderTargetLoadAction colorLoadAction,
  RHIRenderTargetLoadAction /*depthLoadAction*/,
  RHIRenderTargetLoadAction /*stencilLoadAction*/) {
  if (m_currentFrame.passActive) {
    fprintf(stderr, "RHIVK::beginRenderPass: previous render pass not closed\n");
    abort();
  }

  ensureFrameCBOpen();

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

    m_currentFrame.passImage = frame.swapchainImage;
    m_currentFrame.passImageView = swapImageViewFor(frame.swapchainImage, frame.format);
    m_currentFrame.passExtent = frame.extent;
    m_currentFrame.passFormat = frame.format;
  } else {
    // Offscreen RT — Phase 4 supports single-color-attachment, no depth.
    auto* offRT = static_cast<RHIRenderTargetVK*>(target.get());
    if (offRT->colorTargetCount() == 0) {
      fprintf(stderr, "RHIVK::beginRenderPass: offscreen RT with no color targets not supported yet\n");
      abort();
    }
    RHISurfaceVK* color0 = offRT->colorAttachment(0);
    m_currentFrame.passImage = color0->vkImage();
    m_currentFrame.passImageView = color0->vkImageView();
    m_currentFrame.passExtent = vk::Extent2D{color0->width(), color0->height()};
    m_currentFrame.passFormat = color0->vkFormat();
  }
  m_currentFrame.passActive = true;
  m_currentFrame.passIsWindowRT = isWindow;

  vk::CommandBuffer cb = m_currentFrame.commandBuffer;

  // Transition image: UNDEFINED → COLOR_ATTACHMENT_OPTIMAL. UNDEFINED is
  // fine because the load op (clear) writes every pixel. For kLoadKeep
  // we'd need per-image layout tracking — TODO when a use surfaces.
  vk::ImageMemoryBarrier toColorAttachment{
    vk::AccessFlags(),
    vk::AccessFlagBits::eColorAttachmentWrite,
    vk::ImageLayout::eUndefined,
    vk::ImageLayout::eColorAttachmentOptimal,
    VK_QUEUE_FAMILY_IGNORED,
    VK_QUEUE_FAMILY_IGNORED,
    m_currentFrame.passImage,
    vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };
  cb.pipelineBarrier(
    vk::PipelineStageFlagBits::eTopOfPipe,
    vk::PipelineStageFlagBits::eColorAttachmentOutput,
    vk::DependencyFlags(), 0, nullptr, 0, nullptr,
    1, &toColorAttachment);

  // Begin dynamic rendering with clear value from setClearColor.
  vk::ClearValue clearValue{};
  clearValue.color.float32[0] = m_clearColor.r;
  clearValue.color.float32[1] = m_clearColor.g;
  clearValue.color.float32[2] = m_clearColor.b;
  clearValue.color.float32[3] = m_clearColor.a;

  vk::RenderingAttachmentInfoKHR colorAttachment{
    m_currentFrame.passImageView,
    vk::ImageLayout::eColorAttachmentOptimal,
    vk::ResolveModeFlagBits::eNone,
    {},
    vk::ImageLayout::eUndefined,
    (colorLoadAction == kLoadClear) ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad,
    vk::AttachmentStoreOp::eStore,
    clearValue,
  };

  vk::RenderingInfoKHR renderingInfo{
    vk::RenderingFlags(),
    vk::Rect2D{vk::Offset2D{0, 0}, m_currentFrame.passExtent},
    1, // layerCount
    0, // viewMask
    1, // colorAttachmentCount
    &colorAttachment,
  };

  cb.beginRenderingKHR(renderingInfo);
}

void RHIVK::endRenderPass(RHIRenderTarget::ptr /*target*/) {
  if (!m_currentFrame.passActive) {
    fprintf(stderr, "RHIVK::endRenderPass: no render pass in progress\n");
    abort();
  }
  vk::CommandBuffer cb = m_currentFrame.commandBuffer;
  cb.endRenderingKHR();

  // Final layout depends on RT type: present for window, shader-read for
  // offscreen (so the result can be sampled by a subsequent pass in the
  // same CB — pipeline barrier orders the writes/reads).
  vk::ImageLayout finalLayout = m_currentFrame.passIsWindowRT
    ? vk::ImageLayout::ePresentSrcKHR
    : vk::ImageLayout::eShaderReadOnlyOptimal;
  vk::AccessFlags dstAccess = m_currentFrame.passIsWindowRT
    ? vk::AccessFlags()
    : vk::AccessFlagBits::eShaderRead;
  vk::PipelineStageFlags dstStage = m_currentFrame.passIsWindowRT
    ? vk::PipelineStageFlagBits::eBottomOfPipe
    : vk::PipelineStageFlagBits::eFragmentShader;

  vk::ImageMemoryBarrier toFinal{
    vk::AccessFlagBits::eColorAttachmentWrite,
    dstAccess,
    vk::ImageLayout::eColorAttachmentOptimal,
    finalLayout,
    VK_QUEUE_FAMILY_IGNORED,
    VK_QUEUE_FAMILY_IGNORED,
    m_currentFrame.passImage,
    vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };
  cb.pipelineBarrier(
    vk::PipelineStageFlagBits::eColorAttachmentOutput,
    dstStage,
    vk::DependencyFlags(), 0, nullptr, 0, nullptr,
    1, &toFinal);

  // Pass scope done. The CB stays open for the next begin* (next render
  // pass, timer query, etc.); swapBuffers / flush handles the submit.
  m_currentFrame.passActive = false;
  m_currentFrame.passIsWindowRT = false;
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
  m_currentFrame.commandBuffer.setViewportWithCountEXT(1, &vp);
}
void RHIVK::setViewports(const RHIRect* rects, size_t count) {
  if (!m_currentFrame.passActive || count == 0) return;
  std::vector<vk::Viewport> vps(count);
  for (size_t i = 0; i < count; ++i) {
    vps[i] = vk::Viewport{
      float(rects[i].x), float(rects[i].y), float(rects[i].width), float(rects[i].height),
      0.0f, 1.0f};
  }
  m_currentFrame.commandBuffer.setViewportWithCountEXT(uint32_t(count), vps.data());
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

  // Set every dynamic state the implementation requires when shader_object
  // is in use; per-draw setX/bindXState overwrites individual pieces.
  setDynamicStateDefaults();

  // Pipeline-derived dynamic state.
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
  cb.setStencilTestEnableEXT(VK_FALSE);
  cb.setStencilOpEXT(vk::StencilFaceFlagBits::eFrontAndBack,
    vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::StencilOp::eKeep, vk::CompareOp::eAlways);
  cb.setStencilCompareMask(vk::StencilFaceFlagBits::eFrontAndBack, 0xff);
  cb.setStencilWriteMask(vk::StencilFaceFlagBits::eFrontAndBack, 0xff);
  cb.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, 0);

  // Color blend — single color attachment, blending disabled, all channels write.
  const vk::Bool32 blendEnable = VK_FALSE;
  const vk::ColorBlendEquationEXT blendEq{
    vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,
    vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd};
  const vk::ColorComponentFlags writeAll =
    vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
    vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
  cb.setColorBlendEnableEXT(0, 1, &blendEnable);
  cb.setColorBlendEquationEXT(0, 1, &blendEq);
  cb.setColorWriteMaskEXT(0, 1, &writeAll);

  // Default viewport + scissor cover the whole render target.
  vk::Viewport vp{0.0f, 0.0f, float(m_currentFrame.passExtent.width), float(m_currentFrame.passExtent.height), 0.0f, 1.0f};
  cb.setViewportWithCountEXT(1, &vp);
  vk::Rect2D scissor{
    {0, 0},
    m_currentFrame.passExtent
  };
  cb.setScissorWithCountEXT(1, &scissor);
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

  // Engine emits the same blend state for every color attachment when only
  // a single element is provided. Replicate across the bound RT's color
  // attachment count — Phase 4 uses 1 always, so emit just slot 0.
  if (d.targetBlendStates.empty()) return;
  const auto& el = d.targetBlendStates[0];
  const vk::Bool32 enable = el.blendEnabled ? VK_TRUE : VK_FALSE;
  vk::ColorBlendEquationEXT eq{
    vkBlendFactorFor(el.colorSource), vkBlendFactorFor(el.colorDest), vkBlendOpFor(el.colorFunc),
    vkBlendFactorFor(el.alphaSource), vkBlendFactorFor(el.alphaDest), vkBlendOpFor(el.alphaFunc)};
  cb.setColorBlendEnableEXT(0, 1, &enable);
  cb.setColorBlendEquationEXT(0, 1, &eq);
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
  m_currentFrame.commandBuffer.setScissorWithCountEXT(1, &scissor);
}
void RHIVK::clearScissorRect() {
  if (!m_currentFrame.passActive) return;
  vk::Rect2D scissor{
    {0, 0},
    m_currentFrame.passExtent
  };
  m_currentFrame.commandBuffer.setScissorWithCountEXT(1, &scissor);
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

  StagedDescriptorWrite w;
  w.set = it->second.set;
  w.bufferInfo.buffer = static_cast<RHIBufferVK*>(buffer.get())->vkBuffer();
  w.bufferInfo.offset = 0;
  w.bufferInfo.range = buffer->size();
  w.write.dstBinding = it->second.binding;
  w.write.dstArrayElement = 0;
  w.write.descriptorCount = 1;
  w.write.descriptorType = it->second.descriptorType;
  m_pendingDescriptorWrites[{it->second.set, it->second.binding}] = w;
  keepAliveForFrame(buffer);
}

void RHIVK::loadUniformBlockImmediate(FxAtomicString name, const void* data, size_t size) {
  if (!m_boundPipeline) return;
  const auto& bindings = m_boundPipeline->resourceBindings();
  auto it = bindings.find(name.c_str());
  if (it == bindings.end()) return;

  auto alloc = allocTransientUniform(data, size);
  StagedDescriptorWrite w;
  w.set = it->second.set;
  w.bufferInfo.buffer = alloc.buffer;
  w.bufferInfo.offset = alloc.offset;
  w.bufferInfo.range = alloc.size;
  w.write.dstBinding = it->second.binding;
  w.write.dstArrayElement = 0;
  w.write.descriptorCount = 1;
  w.write.descriptorType = it->second.descriptorType;
  m_pendingDescriptorWrites[{it->second.set, it->second.binding}] = w;
}

void RHIVK::loadShaderBuffer(FxAtomicString name, RHIBuffer::ptr buffer) {
  if (!m_boundPipeline) return;
  const auto& bindings = m_boundPipeline->resourceBindings();
  auto it = bindings.find(name.c_str());
  if (it == bindings.end()) return;

  StagedDescriptorWrite w;
  w.set = it->second.set;
  w.bufferInfo.buffer = static_cast<RHIBufferVK*>(buffer.get())->vkBuffer();
  w.bufferInfo.offset = 0;
  w.bufferInfo.range = buffer->size();
  w.write.dstBinding = it->second.binding;
  w.write.dstArrayElement = 0;
  w.write.descriptorCount = 1;
  w.write.descriptorType = it->second.descriptorType;
  m_pendingDescriptorWrites[{it->second.set, it->second.binding}] = w;
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

  StagedDescriptorWrite w;
  w.set = it->second.set;
  w.imageInfo.imageView = static_cast<RHISurfaceVK*>(tex.get())->vkImageView();
  w.imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
  // VK CIS descriptors require a real sampler handle. Callers that pass
  // sampler=nullptr (mirroring GL where the texture's own glTexParameter
  // state suffices) get a process-wide default linear/repeat sampler.
  w.imageInfo.sampler = sampler
    ? static_cast<RHISamplerVK*>(sampler.get())->vkSampler()
    : defaultSampler();
  w.write.dstBinding = it->second.binding;
  w.write.dstArrayElement = 0;
  w.write.descriptorCount = 1;
  w.write.descriptorType = it->second.descriptorType;
  m_pendingDescriptorWrites[{it->second.set, it->second.binding}] = w;
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

  StagedDescriptorWrite w;
  w.set = it->second.set;
  w.imageInfo.imageView = static_cast<RHISurfaceVK*>(tex.get())->vkImageView();
  w.imageInfo.imageLayout = vk::ImageLayout::eGeneral;
  w.write.dstBinding = it->second.binding;
  w.write.dstArrayElement = 0;
  w.write.descriptorCount = 1;
  w.write.descriptorType = it->second.descriptorType;
  m_pendingDescriptorWrites[{it->second.set, it->second.binding}] = w;
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
  vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  vk::SubmitInfo submit;
  submit.setWaitSemaphores(m_currentFrame.imageAcquired);
  submit.setWaitDstStageMask(waitStage);
  submit.setCommandBuffers(cb);
  submit.setSignalSemaphores(m_currentFrame.renderFinished);
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

  // Advance the slot ring.
  m_frameContextIndex = (m_frameContextIndex + 1) % m_frameContexts.size();
  m_currentFrame = {}; // cbOpen=false, hasWindowImage=false, passActive=false
  m_boundPipeline = nullptr;
  m_pendingStreamBuffers.clear();
  m_pendingDescriptorWrites.clear();
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
  vk::SubmitInfo submit;
  submit.setCommandBuffers(cb);
  rhi()->vk()->queue().submit(submit, m_currentFrame.frameFence);

  m_frameContextIndex = (m_frameContextIndex + 1) % m_frameContexts.size();
  m_currentFrame = {};
  m_boundPipeline = nullptr;
  m_pendingStreamBuffers.clear();
  m_pendingDescriptorWrites.clear();
}
void RHIVK::flushAndWaitForGPUScheduling() { RHI_VK_NOT_IMPLEMENTED(); }
uint32_t RHIVK::maxMultisampleSamples() { RHI_VK_NOT_IMPLEMENTED(); }
bool RHIVK::supportsGeometryShaders() { RHI_VK_NOT_IMPLEMENTED(); }

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

// -------- Top-level VK backend init --------

void initRHIVulkan() {
  // No GL context to UUID-match against; select the first enumerated device.
  std::array<uint8_t, VK_UUID_SIZE> uuid{};
  if (!rhiVulkanInstallSingleton(uuid)) {
    fprintf(stderr, "initRHIVulkan: RHIVulkan::create() failed; cannot bring up VK RHI backend\n");
    abort();
  }
  initRHI(new RHIVK());
}
