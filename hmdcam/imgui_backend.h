#pragma once
#include "imgui.h"
#include "rhi/RHIRenderTarget.h"
void ImGui_ImplInputListener_Init();
void ImGui_ImplInputListener_NewFrame();

void ImGui_ImplFxRHI_Init();
void ImGui_ImplFxRHI_NewFrame();
void ImGui_ImplFxRHI_RenderDrawData(RHIRenderTarget::ptr renderTarget, ImDrawData* draw_data);

