#pragma once
#include "rhi/RHIObject.h"
#include "rhi/RHIRenderTarget.h"
#include "rhi/RHISurface.h"
#include <vector>

class RHISwapchain : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHISwapchain> ptr;
  RHISwapchain();
  virtual ~RHISwapchain();

  void addSurface(RHISurface::ptr, RHIRenderTarget::ptr rt = RHIRenderTarget::ptr());
  void advance();

  RHISurface::ptr surface() const { return m_surfaces[m_index]; }
  RHIRenderTarget::ptr renderTarget() const { return m_renderTargets[m_index]; }

protected:
  size_t m_index;
  std::vector<RHISurface::ptr> m_surfaces;
  std::vector<RHIRenderTarget::ptr> m_renderTargets;
};

