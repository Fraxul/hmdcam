#include "V4L2Camera.h"
#include "common/Timing.h"

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("usage: %s video-device\n", argv[0]);
    return -1;
  }

  V4L2Camera* cam = new V4L2Camera(argv[1]);
  printf("Opening camera %s\n", argv[1]);
  if (!cam->tryOpenSensor()) {
    printf("Couldn't open sensor\n");
    return -1;
  }
  


  uint64_t prevTimestamp = 0;
  for (size_t frameIdx = 0; frameIdx < 30; ++frameIdx) {
    if (!cam->readFrame()) {
      printf("readFrame() returned false\n");
      delayNs(1'000'000'000ULL);
      continue;
    }

    printf("sensor timestamp: %lu, delta: %lu\n", cam->sensorTimestamp(), cam->sensorTimestamp() - prevTimestamp);
    prevTimestamp = cam->sensorTimestamp();
  }


  printf("Closing camera\n");
  delete cam;

  return 0;
}
