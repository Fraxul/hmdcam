#include "CANBus.h"
#include "PowerState.h"
#include "GloveControllerPacket.h"
#include <unistd.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
  canbus()->addMessageSubscription(PowerState::kPortId, [](SerializationBuffer& b, const CanardTransferMetadata& md, uint64_t timestamp_sec) {
    PowerState ps;
    ps.handleMessage(b, md, timestamp_sec);
    char buf[256];
    ps.toString(buf, 256);
    printf("%s\n", buf);
  });

  canbus()->addMessageSubscription(GloveControllerPacket::kLeftPortId, [](SerializationBuffer& b, const CanardTransferMetadata& md, uint64_t timestamp_sec) {
    GloveControllerPacket pkt;
    pkt.handleMessage(b, md, timestamp_sec);
    char buf[256];
    pkt.toString(buf, 256);
    printf("[L] %s\n", buf);
  });

  canbus()->addMessageSubscription(GloveControllerPacket::kRightPortId, [](SerializationBuffer& b, const CanardTransferMetadata& md, uint64_t timestamp_sec) {
    GloveControllerPacket pkt;
    pkt.handleMessage(b, md, timestamp_sec);
    char buf[256];
    pkt.toString(buf, 256);
    printf("[R] %s\n", buf);
  });

  while (true) {
    sleep(1);
  }
}
