#pragma once
// RHI-backed ImGui renderer. Mirrors the shape of the upstream
// imgui_impl_*_Init / NewFrame / RenderDrawData backends, but draws through
// the RHI abstraction instead of binding to a specific graphics API.
//
// Shared between hmdcam (HMD presentation path) and debug-client (SDL window
// presentation path). Both apps drive ImGui input from a platform-specific
// source: hmdcam from its InputListener (media remote), debug-client from
// ImGui_ImplSDL2_*. Only the renderer side is shared here.

#include "imgui.h"
#include "rhi/RHIRenderTarget.h"

void ImGui_ImplFxRHI_Init();
void ImGui_ImplFxRHI_NewFrame();
void ImGui_ImplFxRHI_RenderDrawData(RHIRenderTarget::ptr renderTarget, ImDrawData* draw_data);
// Release the static font atlas / blend state / pipeline before the
// underlying RHI device is destroyed. Call after the last RenderDrawData
// and before shutdownRHI().
void ImGui_ImplFxRHI_Shutdown();
