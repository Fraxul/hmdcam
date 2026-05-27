#include "rhi/vk/RHIRenderTargetVK.h"
#include <cstdio>
#include <cstdlib>

/*static*/ RHIRenderTargetVK::ptr RHIRenderTargetVK::compile(const RHIRenderTargetDescriptor& descriptor) {
  if (descriptor.colorTargets.empty() && !descriptor.depthStencilTarget.surface) {
    fprintf(stderr, "RHIRenderTargetVK::compile: descriptor has no attachments\n");
    abort();
  }

  RHIRenderTargetVK::ptr rt(new RHIRenderTargetVK());

  for (const auto& el : descriptor.colorTargets) {
    if (el.singleLayer || el.layerIndex != 0 || el.level != 0) {
      fprintf(stderr, "RHIRenderTargetVK::compile: per-layer / per-mip attachments not supported yet\n");
      abort();
    }
    RHISurfaceVK::ptr s(static_cast<RHISurfaceVK*>(el.surface.get()));
    rt->m_colorAttachments.push_back(s);
    if (rt->m_width == 0) {
      rt->m_width = s->width();
      rt->m_height = s->height();
    }
  }
  if (descriptor.depthStencilTarget.surface) {
    rt->m_depthStencilAttachment = RHISurfaceVK::ptr(static_cast<RHISurfaceVK*>(descriptor.depthStencilTarget.surface.get()));
    if (rt->m_width == 0) {
      rt->m_width = rt->m_depthStencilAttachment->width();
      rt->m_height = rt->m_depthStencilAttachment->height();
    }
  }
  return rt;
}
