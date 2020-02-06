#define IMGUI_IMPL_OPENGL_ES3
#include "imgui/examples/imgui_impl_opengl3.cpp"
#include "InputListener.h"


void ImGui_ImplInputListener_Init() {
  ImGuiIO& io = ImGui::GetIO();

  io.KeyMap[ImGuiKey_UpArrow]    = kButtonUp;
  io.KeyMap[ImGuiKey_DownArrow]  = kButtonDown;
  io.KeyMap[ImGuiKey_LeftArrow]  = kButtonLeft;
  io.KeyMap[ImGuiKey_RightArrow] = kButtonRight;
  io.KeyMap[ImGuiKey_Space]      = kButtonOK;
  io.KeyMap[ImGuiKey_Escape]     = kButtonBack;
}

void ImGui_ImplInputListener_NewFrame() {
  ImGuiIO& io = ImGui::GetIO();

#define IMGUI_KEY(key) io.KeysDown[key] = testButton(key);
  IMGUI_KEY(kButtonUp);
  IMGUI_KEY(kButtonDown);
  IMGUI_KEY(kButtonLeft);
  IMGUI_KEY(kButtonRight);
  IMGUI_KEY(kButtonOK);
  IMGUI_KEY(kButtonBack);

#undef IMGUI_KEY

}

