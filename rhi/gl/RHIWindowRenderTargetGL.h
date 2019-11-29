#pragma once
#include "rhi/gl/RHIRenderTargetGL.h"

class RHIWindowRenderTargetGL : public RHIRenderTargetGL {
public:
  typedef boost::intrusive_ptr<RHIWindowRenderTargetGL> ptr;

  virtual ~RHIWindowRenderTargetGL();

  virtual bool isWindowRenderTarget() const;
  void platformSetUpdatedWindowDimensions(uint32_t x, uint32_t y);
  virtual void platformSwapBuffers() = 0;
protected:
  RHIWindowRenderTargetGL();
};

