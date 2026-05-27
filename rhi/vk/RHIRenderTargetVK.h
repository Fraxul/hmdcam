#pragma once
// RHIRenderTargetVK: offscreen render target built from one or more
// RHISurfaceVK color attachments + optional depth-stencil.
//
// We use ynamic rendering, so there's no VkFramebuffer to bake; the
// render target is just a collection of image views, dimensions, and
// per-attachment surface refs. beginRenderPass reads the views directly.
// Single-layer / single-mip-level surfaces only for now.

#include "rhi/RHIRenderTarget.h"
#include "rhi/vk/RHISurfaceVK.h"
#include <vector>

class RHIRenderTargetVK : public RHIRenderTarget {
public:
  typedef boost::intrusive_ptr<RHIRenderTargetVK> ptr;

  static RHIRenderTargetVK::ptr compile(const RHIRenderTargetDescriptor&);

  virtual ~RHIRenderTargetVK() {}

  virtual uint32_t width() const override { return m_width; }
  virtual uint32_t height() const override { return m_height; }
  virtual uint32_t layers() const override { return 1; }
  virtual uint32_t samples() const override { return 1; }
  virtual bool isArray() const override { return false; }
  virtual bool isWindowRenderTarget() const override { return false; }
  virtual size_t colorTargetCount() const override { return m_colorAttachments.size(); }
  virtual bool hasDepthStencilTarget() const override { return m_depthStencilAttachment != nullptr; }

  RHISurfaceVK* colorAttachment(size_t i) const { return m_colorAttachments[i].get(); }
  RHISurfaceVK* depthStencilAttachment() const { return m_depthStencilAttachment.get(); }

protected:
  RHIRenderTargetVK() = default;

  uint32_t m_width = 0;
  uint32_t m_height = 0;
  std::vector<RHISurfaceVK::ptr> m_colorAttachments;
  RHISurfaceVK::ptr m_depthStencilAttachment;
};
