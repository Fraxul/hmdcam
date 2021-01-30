#pragma once
#include "rhi/RHIConstants.h"
#include "rhi/RHIObject.h"
#include "rhi/RHISurface.h"
#include <stddef.h>
#include <stdint.h>
#include <glm/glm.hpp>
#include <initializer_list>
#include <boost/container/static_vector.hpp>

class RHIRenderTarget : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIRenderTarget> ptr;
  RHIRenderTarget();
  virtual ~RHIRenderTarget();

  virtual uint32_t width() const = 0;
  virtual uint32_t height() const = 0;
  virtual uint32_t layers() const = 0;
  virtual uint32_t samples() const = 0;
  virtual bool isArray() const = 0;

  virtual bool isWindowRenderTarget() const = 0;
  virtual size_t colorTargetCount() const = 0;
  virtual bool hasDepthStencilTarget() const = 0;

  bool hasColorTarget() const { return colorTargetCount() != 0; }
  bool isMultisampled() const { return samples() > 1; }

  bool isStereo() const { return m_isStereo; }
  void setIsStereo(bool value) { m_isStereo = value; }

  glm::vec2 dimensions() const { return glm::vec2(width(), height()); }
  float aspectRatio() const { return static_cast<float>(width()) / static_cast<float>(height()); }

protected:
  bool m_isStereo;
};

struct RHIRenderTargetDescriptorElement {
  RHIRenderTargetDescriptorElement(RHISurface::ptr surface_ = RHISurface::ptr()) : surface(surface_), singleLayer(false), layerIndex(0) {}

  static RHIRenderTargetDescriptorElement singleLayerElement(RHISurface::ptr surface_, uint8_t layerIndex_) {
    RHIRenderTargetDescriptorElement res(surface_);
    res.singleLayer = true;
    res.layerIndex = layerIndex_;
    return res;
  }

  RHISurface::ptr surface;
  bool singleLayer;
  uint8_t layerIndex;
};

struct RHIRenderTargetDescriptor {
  RHIRenderTargetDescriptor() {}

  RHIRenderTargetDescriptor(const std::initializer_list<RHIRenderTargetDescriptorElement>& colorTargets_, RHIRenderTargetDescriptorElement depthStencilTarget_ = RHIRenderTargetDescriptorElement()) : colorTargets(colorTargets_.begin(), colorTargets_.end()), depthStencilTarget(depthStencilTarget_) {}
  RHIRenderTargetDescriptor(const std::initializer_list<RHISurface::ptr>& colorTargets_, RHISurface::ptr depthStencilTarget_ = RHISurface::ptr()) : depthStencilTarget(depthStencilTarget_) {
    for (std::initializer_list<RHISurface::ptr>::const_iterator it = colorTargets_.begin(); it != colorTargets_.end(); ++it) {
      colorTargets.push_back(RHIRenderTargetDescriptorElement(*it));
    }
  }

  boost::container::static_vector<RHIRenderTargetDescriptorElement, kRHIMaxColorRenderTargets> colorTargets;
  RHIRenderTargetDescriptorElement depthStencilTarget;
};

