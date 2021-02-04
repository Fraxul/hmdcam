#pragma once
#include <stddef.h>
#include "rhi/RHISurface.h"

class ICameraProvider {
public:
  virtual ~ICameraProvider() {}

  virtual size_t streamCount() const = 0;
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const = 0;
  virtual unsigned int streamWidth() const = 0;
  virtual unsigned int streamHeight() const = 0;
};

