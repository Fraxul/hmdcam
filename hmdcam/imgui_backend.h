#pragma once
#define IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#define IMGUI_IMPL_OPENGL_ES3
#include "imgui.h"
#include "imgui/backends/imgui_impl_opengl3.h"
void ImGui_ImplInputListener_Init();
void ImGui_ImplInputListener_NewFrame();

