#include "PieMenu.h"
#include "imgui.h"


void GestureMenuTick() {

  //if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(1)) {
  // Handle gestures not intersecting with any window(s)
  if (ImGui::GetIO().WantCaptureMouse == false && ImGui::IsMouseClicked(1)) {
    ImGui::OpenPopup("PieMenu");
  }

  if (BeginPiePopup("PieMenu", /*iMouseButton=*/ 1)) {
    if (PieMenuItem("Test1")) { /*TODO*/ }
    if (PieMenuItem("Test2")) { /*TODO*/ }

    if (PieMenuItem("Test3", false))  { /*TODO*/ }

    if (BeginPieMenu("Sub")) {
      if (BeginPieMenu("Sub sub\nmenu")) {
        if (PieMenuItem("SubSub")) { /*TODO*/ }
        if (PieMenuItem("SubSub2")) { /*TODO*/ }
        EndPieMenu();
      }
      if (PieMenuItem("TestSub")) { /*TODO*/ }
      if (PieMenuItem("TestSub2")) { /*TODO*/ }
      EndPieMenu();
    }

    EndPiePopup();
  }
}

