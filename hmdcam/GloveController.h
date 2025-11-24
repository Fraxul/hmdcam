#pragma once
#include "GloveControllerPacket.h"
#include "common/ScrollingBuffer.h"
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

    // Each received packet pushes the delta-time between the previous and current packets to this.
    // Good for catching latency jitter.
    ScrollingBuffer<float> packetDeltaTimesMs;

    uint64_t lastPacketTimestampNs = 0; // currentTimeNs()
    float lastPacketAgeMs() const { return deltaTimeMs(lastPacketTimestampNs, currentTimeNs()); }

    glm::vec3 acceleration_g = glm::vec3(0.0f); // G
    glm::vec3 rotation_dps = glm::vec3(0.0f); // degrees/sec
    uint8_t buttonState = 0;


    uint8_t previousButtonState = 0; // Used to only submit button deltas to imgui

    uint8_t releasedButtonsForPacketLossMask = 0;
  };

  ControllerState m_controllerState[kMaxControllerId];

  float m_gyroDeadzoneDPS = 1.0f;
  float m_gyroLog2Gain = -3.3f;
};


