#include "imgui_backend.h"
#include "InputListener.h"

void ImGui_ImplInputListener_Init() {
  // ImGuiIO& io = ImGui::GetIO();
}

void ImGui_ImplInputListener_NewFrame() {
  ImGuiIO& io = ImGui::GetIO();

  io.AddKeyEvent(ImGuiKey_Menu, testButton(kButtonPower));

  io.AddKeyEvent(ImGuiKey_UpArrow, testButton(kButtonUp));
  io.AddKeyEvent(ImGuiKey_DownArrow, testButton(kButtonDown));
  io.AddKeyEvent(ImGuiKey_LeftArrow, testButton(kButtonLeft));
  io.AddKeyEvent(ImGuiKey_RightArrow, testButton(kButtonRight));
  io.AddKeyEvent(ImGuiKey_Space, testButton(kButtonOK));
  io.AddKeyEvent(ImGuiKey_Escape, testButton(kButtonBack));
}
