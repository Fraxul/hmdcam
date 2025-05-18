#include "V4L2Camera.h"
#include "common/Timing.h"
#include <signal.h>

bool want_quit = false;
static void signal_handler(int) {
  want_quit = true;

  // Restore signal handlers so the program is still interruptable if clean shutdown gets stuck
  signal(SIGINT,  SIG_DFL);
  signal(SIGTERM, SIG_DFL);
  signal(SIGQUIT, SIG_DFL);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    printf("usage: %s video-device\n", argv[0]);
    return -1;
  }

  V4L2Camera* cam = new V4L2Camera();
  printf("Opening camera %s\n", argv[1]);

  if (!cam->tryOpenSensor(argv[1])) {
    printf("Couldn't open sensor\n");
    return -1;
  }

  signal(SIGINT,  signal_handler);
  signal(SIGTERM, signal_handler);
  signal(SIGQUIT, signal_handler);

  uint64_t prevTimestamp = 0;
  while(!want_quit) {
    if (!cam->readFrame()) {

      if (want_quit)
        break; // readFrame() probably failed due to EINTR from ctrl-c

      printf("readFrame() returned false\n");
      delayNs(1'000'000'000ULL);

      while (!cam->tryOpenSensor(argv[1])) {
        if (want_quit)
          break;

        printf("Couldn't open sensor\n");
        delayNs(1'000'000'000ULL);
      }
      continue;
    }

    printf("sensor timestamp: %lu, delta: %lu\n", cam->sensorTimestamp(), cam->sensorTimestamp() - prevTimestamp);
    prevTimestamp = cam->sensorTimestamp();
  }


  printf("Closing camera\n");
  delete cam;

  return 0;
}
