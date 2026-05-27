#include "rhi/vk/RHIWindowRenderTargetVK.h"

RHIWindowRenderTargetVK::RHIWindowRenderTargetVK(uint32_t width, uint32_t height, vk::Format format) :
  m_width(width),
  m_height(height),
  m_format(format) {
}

RHIWindowRenderTargetVK::~RHIWindowRenderTargetVK() {
}
