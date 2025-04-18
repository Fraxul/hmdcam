#pragma once
#include "Render.h"

void RenderInitDebugSurface(uint32_t width, uint32_t height);
bool RenderDebugSubsystemEnabled();
RHISurface::ptr renderAcquireDebugSurface();
void renderSubmitDebugSurface(RHISurface::ptr);
const char* renderDebugURL();

