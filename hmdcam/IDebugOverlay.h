#pragma once
#include <rhi/RHISurface.h>

enum DebugOverlayType {
  kDebugOverlayNone,
  kDebugOverlayFocusAssist,
};

class IDebugOverlay {
public:
  IDebugOverlay() = default;
  virtual ~IDebugOverlay() {}

  virtual DebugOverlayType overlayType() const = 0;

  virtual void update() = 0;
  virtual void renderIMGUI() = 0;
  virtual RHISurface::ptr overlaySurfaceForCamera(size_t cameraIdx) = 0;

private:
  IDebugOverlay& operator=(const IDebugOverlay&) = delete;
  IDebugOverlay(const IDebugOverlay&) = delete;
};

