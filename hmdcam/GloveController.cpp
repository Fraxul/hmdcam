#include "GloveController.h"
#include "CANBus.h"
#include "GloveControllerPacket.h"
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include "imgui_backend.h"


GloveController::GloveController() {
  // Subscribe to CAN packets from our controllers
  canbus()->addMessageSubscription(GloveControllerPacket::kLeftPortId, [this](SerializationBuffer& b, const CanardTransferMetadata& md, uint64_t timestamp_sec) {
    GloveControllerPacket pkt;
    pkt.handleMessage(b, md, timestamp_sec);
    if (pkt.valid()) {
      handleControllerPacket(kLeftController, pkt);
    }
  });

  canbus()->addMessageSubscription(GloveControllerPacket::kRightPortId, [this](SerializationBuffer& b, const CanardTransferMetadata& md, uint64_t timestamp_sec) {
    GloveControllerPacket pkt;
    pkt.handleMessage(b, md, timestamp_sec);
    if (pkt.valid()) {
      handleControllerPacket(kRightController, pkt);
    }
  });

  ImGui::GetIO().MouseDrawCursor = true;
}

void GloveController::handleControllerPacket(ControllerId controllerId, const GloveControllerPacket& packet) {
  assert(controllerId < kMaxControllerId);

  ControllerState& state = m_controllerState[controllerId];
  state.acceleration_g = glm::vec3(
    static_cast<float>(packet.accel_milliG[0]) / 1000.0f,
    static_cast<float>(packet.accel_milliG[1]) / 1000.0f,
    static_cast<float>(packet.accel_milliG[2]) / 1000.0f);
  state.rotation_dps = glm::vec3(
    static_cast<float>(packet.gyro_milliDPS[0]) / 1000.0f,
    static_cast<float>(packet.gyro_milliDPS[1]) / 1000.0f,
    static_cast<float>(packet.gyro_milliDPS[2]) / 1000.0f);
  state.buttonState = packet.buttonState;
  state.lastPacketTimestampNs = currentTimeNs();

#if 0
  char buf[256];
  packet.toString(buf, sizeof(buf));
  printf("[%s] %s\n", controllerId == kLeftController ? "L" : "R", buf);
#endif
}

void GloveController::processFrame() {
  for (uint8_t controllerId = 0; controllerId < kMaxControllerId; ++controllerId) {
    ControllerState& state = m_controllerState[controllerId];

    if (state.lastPacketAgeMs() > 100) {
      // No packets received from controller for a while, so we stop reporting events from it.

      // TODO: If any controller buttons were pressed, we need to release them.

      continue;
    }

    // TODO: Only apply mouse input when the appropriate button is held.


    // Collapse gyroscope axes down to 2d projection.
    // Use the accelerometer vector to rotate around the roll axis to align the mouse Y axis with gravity.
    // Default accelerometer orientation: X is roll, Y is pitch (mouse up/down), Z is yaw (mouse L/R)
    // +Z faces down, +Y faces left, +X faces forward.

    // 2d gravity vector projection against the roll axis
    glm::vec2 gravityRelLeft = glm::normalize(glm::vec2(-state.acceleration_g.z, state.acceleration_g.y));
    glm::vec2 gravityRelDown = glm::vec2(-gravityRelLeft.y, gravityRelLeft.x);

    // Correct for gravity orientation
    glm::vec2 gyroMouseXY = (state.rotation_dps.y * gravityRelDown) + (state.rotation_dps.z * gravityRelLeft);

    // Apply deadzone
    gyroMouseXY = glm::max(glm::abs(gyroMouseXY) - m_gyroDeadzoneDPS, glm::vec2(0.0f)) * glm::sign(gyroMouseXY);

    // Apply scaling
    gyroMouseXY = gyroMouseXY * powf(2.0f, m_gyroLog2Gain);

    if (glm::abs(gyroMouseXY.x) < 1.0f && glm::abs(gyroMouseXY.y) < 1.0f)
      continue; // No significant movement

    // Synthesize mouse events
    auto& io = ImGui::GetIO();

    // AddMousePosEvent accepts absolute coordinates, so add our relative offset to the previous absolute position.
    // Clamp to [0, io.DisplaySize - 1] to stay within the screen bounds.
    glm::vec2 absPosition = glm::clamp(glm::vec2(io.MousePos.x, io.MousePos.y) + gyroMouseXY, glm::vec2(0.0f), glm::vec2(io.DisplaySize.x - 1.0f, io.DisplaySize.y - 1.0f));

    io.AddMousePosEvent(absPosition.x, absPosition.y);
    // printf("[%s] Rel=[%.3f, %.3f] Abs=[%.3f, %.3f]\n", (controllerId == kLeftController) ? "L" : "R", gyroMouseXY.x, gyroMouseXY.y, absPosition.x, absPosition.y);

  }

}

void GloveController::drawConfigIMGUI() {
  ImGui::DragFloat("Deadzone", &m_gyroDeadzoneDPS, /*v_speed=*/ 1.0f, /*v_min=*/ 0.0f, /*v_max=*/ 100.0f, "%.1f", ImGuiSliderFlags_None);
  ImGui::DragFloat("Log2(gain)", &m_gyroLog2Gain, /*v_speed=*/ 0.1f, /*v_min=*/ -10.0f, /*v_max=*/ 10.0f, "%.1f", ImGuiSliderFlags_None);
}

