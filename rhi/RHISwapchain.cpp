#include "rhi/RHISwapchain.h"
#include "rhi/RHI.h"

RHISwapchain::RHISwapchain() : m_index(0) {

}

RHISwapchain::~RHISwapchain() {

}

void RHISwapchain::addSurface(RHISurface::ptr srf, RHIRenderTarget::ptr rt) {
  assert(srf);

  if (!rt) {
    rt = rhi()->compileRenderTarget(RHIRenderTargetDescriptor({srf}));
  }

  m_surfaces.push_back(srf);
  m_renderTargets.push_back(rt);
}

void RHISwapchain::advance() {
  m_index += 1;
  if (m_index >= m_surfaces.size()) {
    m_index = 0;
  }
}

