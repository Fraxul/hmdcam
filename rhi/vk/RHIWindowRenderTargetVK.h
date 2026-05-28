#pragma once
// RHIWindowRenderTargetVK: Vulkan-side window render target.
//
// Lightweight handle that identifies "this is the swapchain window." The
// actual underlying VkImage rotates per-frame as the swap chain advances;
// RHIVK is responsible for binding the currently-acquired image during
// beginRenderPass when this RT is the target.
//
// The recorded width/height/format come from the swapchain at creation time
// and are immutable for the swapchain's lifetime.

#include "rhi/RHIRenderTarget.h"

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#include "vulkan/vulkan.hpp"
#pragma clang diagnostic pop

class RHIWindowRenderTargetVK : public RHIRenderTarget {
public:
  typedef boost::intrusive_ptr<RHIWindowRenderTargetVK> ptr;

  RHIWindowRenderTargetVK(uint32_t width, uint32_t height, vk::Format format);
  virtual ~RHIWindowRenderTargetVK();

  virtual uint32_t width() const override { return m_width; }
  virtual uint32_t height() const override { return m_height; }
  virtual uint32_t layers() const override { return 1; }
  virtual uint32_t samples() const override { return 1; }
  virtual bool isArray() const override { return false; }

  virtual bool isWindowRenderTarget() const override { return true; }
  virtual size_t colorTargetCount() const override { return 1; }
  virtual bool hasDepthStencilTarget() const override { return false; }

  vk::Format format() const { return m_format; }

  // Called by the owning frame source after a swapchain recreation (e.g.
  // window resize). The dimensions/format are queried at next beginRenderPass
  // — there is no internal VkImage state to update.
  void platformSetUpdatedWindowParameters(uint32_t width, uint32_t height, vk::Format format) {
    m_width = width;
    m_height = height;
    m_format = format;
  }

protected:
  uint32_t m_width;
  uint32_t m_height;
  vk::Format m_format;
};
