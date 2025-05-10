#include <stdio.h>
#include "CuDLAStandaloneRunner.h"
#include "../common/Timing.h"

int main(int argc, char* argv[]) {


  if (argc < 3) {
    printf("usage: %s deviceIdx standalone-dla-runnable.engine\n", argv[0]);
    return -1;
  }

  uint64_t deviceIdx;
  {
    char* endPtr = nullptr;
    deviceIdx = strtol(argv[1], &endPtr, 10);
    if (endPtr == argv[1] || deviceIdx > 1) {
      printf("Invalid device index\n");
      return -1;
    }
  }

  CuDLAStandaloneRunner runner(deviceIdx, argv[2]);

  const int runCount = 200;

  while (true) {
    uint64_t startTime = currentTimeNs();
    for (int i = 0; i < runCount; ++i) {
      runner.runInference();
    }
    uint64_t endTime = currentTimeNs();
    printf("%d runs, average %.3f ms/run\n", runCount, deltaTimeMs(startTime, endTime) / static_cast<float>(runCount));
  }
  return 0;
}
