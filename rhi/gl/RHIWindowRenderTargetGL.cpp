#include "rhi/gl/RHIWindowRenderTargetGL.h"

RHIWindowRenderTargetGL::RHIWindowRenderTargetGL() {
  m_glFramebufferId = 0;
  m_width = m_height = 0;
  m_layers = 1;
  m_samples = 1;
  m_colorTargetCount = 1;
  m_isArray = false;
  m_hasDepthStencilTarget = true;
}

RHIWindowRenderTargetGL::~RHIWindowRenderTargetGL() {

}

void RHIWindowRenderTargetGL::platformSetUpdatedWindowDimensions(uint32_t x, uint32_t y) {
  m_width = x;
  m_height = y;
}

bool RHIWindowRenderTargetGL::isWindowRenderTarget() const {
  return true;
}

