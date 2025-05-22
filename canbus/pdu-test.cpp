#include "CANBus.h"
#include "PowerState.h"
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

  while (true) {
    sleep(1);
  }
}
