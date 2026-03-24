#include "PieMenu.h"
#include "imgui.h"
#include "common/DepthMapGenerator.h"
#include "common/Timing.h"
#include <signal.h>
#ifdef IS_TEGRA
#define USE_EYETRACKING
#include "EyeTrackingService.h"
#include "FaceTrackingService.h"
#endif


#ifdef USE_EYETRACKING
extern EyeTrackingService* eyeTrackingService;
extern FaceTrackingService* faceTrackingService;
#endif // USE_EYETRACKING
extern DepthMapGenerator* depthMapGenerator;

void GestureMenuTick() {
  auto& io = ImGui::GetIO();

  //if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(1)) {
  // Handle gestures not intersecting with any window(s)
  if (ImGui::GetIO().WantCaptureMouse == false && ImGui::IsMouseClicked(1)) {
    // Teleport mouse to screen-center before opening the popup.
    io.MousePos = ImVec2(io.DisplaySize.x / 2, io.DisplaySize.y / 2);
    ImGui::OpenPopup("PieMenu");
  }

  if (BeginPiePopup("PieMenu", /*iMouseButton=*/ 1)) {
    if (BeginPieMenu("Process")) {
      // Unlock logic with timeout.
      static uint64_t lastUnlockTimeNs = 0;
      static constexpr float unlockTimeoutMs = (10 * 1000);

      bool unlocked = deltaTimeMs(lastUnlockTimeNs, currentTimeNs()) < unlockTimeoutMs;

      if (unlocked) {
        if (PieMenuItem("Lock\nControls")) {
          lastUnlockTimeNs = 0;
        }
      } else {
        if (PieMenuItem("Unlock\nControls")) {
          lastUnlockTimeNs = currentTimeNs();
        }
      }

      ImGui::BeginDisabled(!unlocked);
      if (PieMenuItem("Terminate\n(exit())")) {
        exit(0);
      }

      if (PieMenuItem("Terminate\n(SIGTERM)")) {
        kill(getpid(), SIGTERM);
        _exit(0); // just in case?
      }

      if (PieMenuItem("Shutdown")) {
        system("sudo shutdown -h now");
        exit(0);
      }

      if (PieMenuItem("Reboot")) {
        system("sudo reboot");
        exit(0);
      }

      ImGui::EndDisabled();
      EndPieMenu();
    }

    if (BeginPieMenu("Render")) {

      if (PieMenuItem("Toggle\nFixed\nDisparity")) {
        if (depthMapGenerator) {
          depthMapGenerator->setDebugUseFixedDisparity(!depthMapGenerator->debugUseFixedDisparity());
        }
      }

      EndPieMenu();
    }

    if (BeginPieMenu("Eye\nTrack")) {
#ifdef USE_EYETRACKING
      // TODO
#endif
      EndPieMenu();
    }

    if (BeginPieMenu("Face\nTrack")) {
#ifdef USE_EYETRACKING
      // TODO
#endif
      EndPieMenu();
    }

#if 0
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
#endif


    EndPiePopup();
  }
}

