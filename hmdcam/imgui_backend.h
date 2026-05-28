#pragma once
#include "imgui.h"
#include "rhi/imgui/RHIImGuiBackend.h"

// hmdcam-specific: bridge ImGui input to the InputListener (media remote +
// keyboard) thread.
void ImGui_ImplInputListener_Init();
void ImGui_ImplInputListener_NewFrame();
