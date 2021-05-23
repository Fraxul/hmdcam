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

#include <vector>

#define NVML_CHECK(x) checkNvmlReturn(x, #x, __FILE__, __LINE__)

// Constants to enable PWM on pin 18 of the AGX Xavier's 40 pin connector.
const size_t pinmux_register = 0x2434090;
const uint32_t pinmux_register_value = 0x401;

#define PWM_BASE "/sys/class/pwm/pwmchip2"

const uint64_t pwmPeriod = 50000; // nanoseconds

bool oneshot_write(const char* path, const char* value) {
  int fd = open(path, O_WRONLY);
  if (fd < 0) {
    fprintf(stderr, "error opening %s: %s\n", path, strerror(errno));
    return false;
  }

  bool res = true;

  if (write(fd, value, strlen(value) + 1) < 0) {
    fprintf(stderr, "error writing to %s (value: \"%s\"): %s\n", path, value, strerror(errno));
    res = false;
  }

  close(fd);
  return res;
}

bool pinmuxSetup() {
  // syscall sequence borrowed from `busybox devmem`
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    fprintf(stderr, "unable to open /dev/mem: %s\n", strerror(errno));
    return false;
  }

  size_t pinmux_page = pinmux_register & (~(4095ULL));
  size_t pinmux_offset = (pinmux_register - pinmux_page) / 4;

  uint32_t* region = static_cast<uint32_t*>(mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, pinmux_page));
  if (!region) {
    fprintf(stderr, "mmap() failed: %s\n", strerror(errno));
    close(fd);
    return false;
  }

  if (region[pinmux_offset] != pinmux_register_value) {
    printf("Setting pinmux register %#zx to %#x (was %#x)\n",
      pinmux_register, pinmux_register_value, region[pinmux_offset]);

    region[pinmux_offset] = pinmux_register_value;
  }

  printf("Pinmux current value: %#x\n", region[pinmux_offset]);

  munmap(region, 4096);
  close(fd);
  return true;
}

void pwmSetup() {
  // no error checking here -- the pwmchip might already be sysfs-exported
  oneshot_write(PWM_BASE "/export", "0");
}

bool pwmSetPeriod() {
  char buf[32];
  sprintf(buf, "%lu", pwmPeriod);
  return oneshot_write(PWM_BASE "/pwm0/period", buf);
}

bool pwmSetEnabled(bool enabled) {
  return oneshot_write(PWM_BASE "/pwm0/enable", enabled ? "1" : "0");
}

// 0-1 range
bool pwmSetDutyCycle(float dutyCycle) {
  dutyCycle = glm::clamp(dutyCycle, 0.0f, 1.0f);

  char buf[32];
  sprintf(buf, "%lu", static_cast<unsigned long>(dutyCycle * static_cast<float>(pwmPeriod)));
  return oneshot_write(PWM_BASE "/pwm0/duty_cycle", buf);
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

  for (int i = 1; i < argc; ++i) {
    if ((!strcmp(argv[i], "--quiet")) || (!strcmp(argv[i], "-q"))) {
      quiet = true;
    } else {
      printf("Unrecognized argument: %s\n", argv[i]);
      return -1;
    }
  }

  if (!pinmuxSetup()) {
    printf("Pinmux setup failed\n");
    return -1;
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

  // open PWM
  pwmSetup();
  pwmSetPeriod();
  pwmSetDutyCycle(0.5f);
  pwmSetEnabled(true);


  while (true) {
    unsigned int coreTemp; // in degrees C
    NVML_CHECK(nvmlDeviceGetTemperature(hDevice, NVML_TEMPERATURE_GPU, &coreTemp));

    // setting lower bound of duty cycle to 20% to keep the fan on
    float dutyCycle = glm::mix(0.2f, 1.0f, glm::smoothstep(30.0f, 70.0f, static_cast<float>(coreTemp)));

    if (!quiet) {
      printf("GPU temp %u => duty cycle %u%%\n", coreTemp, static_cast<unsigned int>(dutyCycle * 100.0f));
    }

    pwmSetDutyCycle(dutyCycle);

    sleep(2);
  }


  nvmlShutdown();
  return 0;
}

