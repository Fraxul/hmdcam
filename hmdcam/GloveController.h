#pragma once
#include "GloveControllerPacket.h"
#include "common/Timing.h"
#include <glm/glm.hpp>

class GloveController : boost::noncopyable {
public:
  GloveController();

  enum ControllerId : uint8_t {
    kLeftController,
    kRightController,
    kMaxControllerId
  };

  void handleControllerPacket(ControllerId controller, const GloveControllerPacket& packet);

  void processFrame();
  void drawConfigIMGUI();

protected:

  struct ControllerState {
    uint64_t lastPacketTimestampNs = 0; // currentTimeNs()
    float lastPacketAgeMs() const { return deltaTimeMs(lastPacketTimestampNs, currentTimeNs()); }

    glm::vec3 acceleration_g = glm::vec3(0.0f); // G
    glm::vec3 rotation_dps = glm::vec3(0.0f); // degrees/sec
    uint8_t buttonState = 0;
  };

  ControllerState m_controllerState[kMaxControllerId];

  float m_gyroDeadzoneDPS = 5.0f;
  float m_gyroLog2Gain = -3.8f;
};


