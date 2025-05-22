#include "PDUComms.h"
#include "CANBus.h"
#include "PowerState.h"
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include "imgui_backend.h"


const size_t kPDUStatusLineSize = 1024;
char pduStatusLine[kPDUStatusLineSize];
PowerState powerState;

void drawPDUStatusLine() {
  if (powerState.valid()) {
    powerState.toString(pduStatusLine, kPDUStatusLineSize);
    ImGui::TextUnformatted(pduStatusLine);
  }
}

void drawPDUCommandMenu() {

}

void startPDUCommsThread() {
  canbus()->addMessageSubscription(PowerState::kPortId, [](SerializationBuffer& b, const CanardTransferMetadata& md, uint64_t timestamp_sec) {
    powerState.handleMessage(b, md, timestamp_sec);
  });
}

