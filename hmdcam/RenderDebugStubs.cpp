#include "Render.h"
#include "RenderDebug.h"

// Linker stubs for the debug subsystem.
// It's assumed that an implementation will override these functions.

__attribute__((weak)) void RenderInitDebugSurface(uint32_t width, uint32_t height) {
}

__attribute__((weak)) bool RenderDebugSubsystemEnabled() {
  return false;
}

__attribute__((weak)) RHISurface::ptr renderAcquireDebugSurface() {
  return RHISurface::ptr();
}

__attribute__((weak)) void renderSubmitDebugSurface(RHISurface::ptr debugSurface) {
}

__attribute__((weak)) const char* renderDebugURL() {
  return nullptr;
}

