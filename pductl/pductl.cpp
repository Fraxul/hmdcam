#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <nvml.h>
#include <glm/glm.hpp>

#include <termios.h>
#include <unistd.h>

#include <vector>
#include <string>

#define NVML_CHECK(x) checkNvmlReturn(x, #x, __FILE__, __LINE__)

int serialFd;

// 0-1 range
bool pwmSetDutyCycle(float dutyCycle) {
  dutyCycle = glm::clamp(dutyCycle, 0.0f, 1.0f);

  char buf[32];
  sprintf(buf, "pwm %u\n", (unsigned int) (dutyCycle * 255.0f));
  if (write(serialFd, buf, strlen(buf)) < 0) {
    fprintf(stderr, "error writing PWM command: %s\n", strerror(errno));
    return false;
  }

  return true;
}

nvmlReturn_t checkNvmlReturn(nvmlReturn_t res, const char* op, const char* file, int line) {
  if (res != NVML_SUCCESS) {
    const char* errorDesc = nvmlErrorString(res);
    fprintf(stderr, "%s (%s:%d) returned nvmlReturn_t %d: %s\n", op, file, line, res, errorDesc);
    abort();
  }
  return res;
}

int main(int argc, char* argv[]) {
  bool quiet = false;
  std::string serialPort = "/dev/ttyTHS0";

  for (int i = 1; i < argc; ++i) {
    if ((!strcmp(argv[i], "--quiet")) || (!strcmp(argv[i], "-q"))) {
      quiet = true;
    } else if (!strcmp(argv[i], "--port")) {
      serialPort = argv[++i];
    } else {
      printf("Unrecognized argument: %s\n", argv[i]);
      return -1;
    }
  }

  serialFd = open(serialPort.c_str(), O_RDWR);
  if (serialFd < 0) {
    fprintf(stderr, "error opening %s: %s\n", serialPort.c_str(), strerror(errno));
    return -1;
  }

  // Set baudrate to 115,200
  {
    struct termios options;
    if (tcgetattr(serialFd, &options) != 0) {
      perror("tcgetattr");
      return -1;
    }
    cfmakeraw(&options);
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);
    if (tcsetattr(serialFd, TCSANOW, &options) != 0) {
      perror("tcsetattr");
      return -1;
    }
  }


  NVML_CHECK(nvmlInit_v2());

  unsigned int deviceCount = 0;
  NVML_CHECK(nvmlDeviceGetCount_v2(&deviceCount));
  if (!deviceCount) {
    printf("nvmlDeviceGetCount_v2 did not return any devices\n");
    return -1;
  }

  std::vector<nvmlDevice_t> deviceHandles(deviceCount);

  printf("Devices (%u):\n", deviceCount);
  for (unsigned int deviceIdx = 0; deviceIdx < deviceCount; ++deviceIdx) {
    char nameBuf[64];
    NVML_CHECK(nvmlDeviceGetHandleByIndex(deviceIdx, &deviceHandles[deviceIdx]));
    NVML_CHECK(nvmlDeviceGetName(deviceHandles[deviceIdx], nameBuf, 64));
    printf("[%u] %s\n", deviceIdx, nameBuf);
  }

  // just using first device
  nvmlDevice_t hDevice = deviceHandles[0];

  // dump temperature thresholds
/*
  unsigned int tempShutdown, tempSlowdown, tempGpuMax; // all in degrees C
  nvmlDeviceGetTemperatureThreshold(hDevice, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, &tempShutdown);
  nvmlDeviceGetTemperatureThreshold(hDevice, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &tempSlowdown);
  nvmlDeviceGetTemperatureThreshold(hDevice, NVML_TEMPERATURE_THRESHOLD_GPU_MAX, &tempGpuMax);
  printf("Device temperature thresholds: shutdown=%u slowdown=%u gpuMax=%u\n",
    tempShutdown, tempSlowdown, tempGpuMax);
*/

  while (true) {
    unsigned int coreTemp; // in degrees C
    NVML_CHECK(nvmlDeviceGetTemperature(hDevice, NVML_TEMPERATURE_GPU, &coreTemp));

    // Fan is good all the way down to 0% duty cycle (it doesn't stop)
    float dutyCycle = glm::mix(0.0f, 1.0f, glm::smoothstep(30.0f, 60.0f, static_cast<float>(coreTemp)));

    if (!quiet) {
      printf("GPU temp %u => duty cycle %u%%\n", coreTemp, static_cast<unsigned int>(dutyCycle * 100.0f));
    }

    pwmSetDutyCycle(dutyCycle);

    sleep(2);
  }


  nvmlShutdown();
  return 0;
}

