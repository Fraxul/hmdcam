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
extern void debugRestartCapture();
extern bool drawStatusBar;

int gestureMenuButton = -1;
void GestureMenuTick() {
  auto& io = ImGui::GetIO();

  //if (ImGui::IsWindowHovered() && ImGui::IsMouseClicked(1)) {
  // Handle gestures not intersecting with any window(s)
  if (gestureMenuButton == -1) {
    // See if any buttons are pressed.
    if (ImGui::GetIO().WantCaptureMouse == false) {
      for (int buttonIdx = 0; buttonIdx < ImGuiMouseButton_COUNT; ++buttonIdx) {
        if (ImGui::IsMouseClicked(buttonIdx)) {
          gestureMenuButton = buttonIdx;
          break;
        }
      }

      if (gestureMenuButton >= 0) {
        // Teleport mouse to screen-center before opening the popup.
        ImVec2 screenCenter = ImVec2(io.DisplaySize.x / 2, io.DisplaySize.y / 2);
        io.MousePos = screenCenter; // Applies this frame
        io.AddMousePosEvent(screenCenter.x, screenCenter.y); // Applies to future frames
        ImGui::OpenPopup("PieMenu");
      }
    }
  } else {
    if (ImGui::IsMouseReleased(gestureMenuButton)) {
      // Button is released this frame, so un-latch from it.
      // BeginPiePopup will accept -1 as "keep using the previous button".
      gestureMenuButton = -1;
    }
  }

  if (BeginPiePopup("PieMenu", gestureMenuButton)) {
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

      if (PieMenuItem(drawStatusBar ? "Hide\nStatus\nBar" : "Show\nStatus\nBar")) {
        drawStatusBar = !drawStatusBar;
      }

      bool useFixedDisparity = depthMapGenerator ? depthMapGenerator->debugUseFixedDisparity() : false;
      if (PieMenuItem(useFixedDisparity ? "Disable\nFixed\nDisparity" : "Enable\nFixed\nDisparity")) {
        if (depthMapGenerator) {
          depthMapGenerator->setDebugUseFixedDisparity(!depthMapGenerator->debugUseFixedDisparity());
        }
      }

      if (PieMenuItem("Restart\nCapture")) {
        debugRestartCapture();
      }

      EndPieMenu();
    }

#ifdef USE_EYETRACKING
    if (eyeTrackingService && BeginPieMenu("Eye\nTrack")) {
      if (PieMenuItem(eyeTrackingService->m_debugShowFeedbackView ? "Hide\nFeedback" : "Show\nFeedback")) {
        eyeTrackingService->m_debugShowFeedbackView = !eyeTrackingService->m_debugShowFeedbackView;
      }

      if (PieMenuItem(eyeTrackingService->m_enableBlink ? "Disable\nBlink" : "Enable\nBlink")) {
        eyeTrackingService->m_enableBlink = !eyeTrackingService->m_enableBlink;
      }

      if (PieMenuItem("Recalibrate")) {
        eyeTrackingService->debugClearCalibration();
      }

      if (PieMenuItem(eyeTrackingService->m_debugSaveBadFitImages ? "Disable\nBad-fit\nImage Save" : "Enable\nBad-fit\nImage Save")) {
        eyeTrackingService->m_debugSaveBadFitImages = !eyeTrackingService->m_debugSaveBadFitImages;
      }

      if (PieMenuItem(eyeTrackingService->m_debugDisableProcessing ? "Enable\nProcessing" : "Disable\nProcessing")) {
        eyeTrackingService->m_debugDisableProcessing = !eyeTrackingService->m_debugDisableProcessing;
      }

      EndPieMenu();
    }

    if (BeginPieMenu("Face\nTrack")) {
      if (PieMenuItem(faceTrackingService->m_debugShowFeedbackView ? "Hide\nFeedback" : "Show\nFeedback")) {
        faceTrackingService->m_debugShowFeedbackView = !faceTrackingService->m_debugShowFeedbackView;
      }

      if (PieMenuItem(faceTrackingService->m_debugDisableProcessing ? "Enable\nProcessing" : "Disable\nProcessing")) {
        faceTrackingService->m_debugDisableProcessing = !faceTrackingService->m_debugDisableProcessing;
      }

      EndPieMenu();
    }
#endif

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

