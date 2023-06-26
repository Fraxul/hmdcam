#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <poll.h>
#include <sys/fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <errno.h>
#include <nvml.h>
#include <glm/glm.hpp>
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>
#include "PDUControl.h"

#include <unistd.h>

#include <vector>
#include <string>

#define NVML_CHECK(x) checkNvmlReturn(x, #x, __FILE__, __LINE__)

PDUControl pduControl;

bool quiet = false;
float minDutyCycle = 0.0f;
float maxDutyCycle = 1.0f;
float minTemp = 30.0f;
float maxTemp = 60.0f;

enum FanID {
  kFanCPU,
  kFanGPU
};

// ^^^\n
// Status Line 1
// Status Line 2
// More status lines...
// $$$\n

static boost::regex infoExpr("\\^\\^\\^[[:space:]]*(.*?)[[:space:]]*\\$\\$\\$");

// 0-1 range
#if 0
bool pwmSetDutyCycle(FanID fanID, float dutyCycle) {
  dutyCycle = glm::clamp(dutyCycle, 0.0f, 1.0f);

  char buf[32];
  sprintf(buf, "pwm %s %u\n", (fanID == kFanCPU) ? "fan1" : "fan2", (unsigned int) (dutyCycle * 255.0f));
  if (write(serialFd, buf, strlen(buf)) < 0) {
    fprintf(stderr, "error writing PWM command: %s\n", strerror(errno));
    return false;
  }

  return true;
}
#endif

float readThermalZoneSensor() {
  const char* sensorFile = "/sys/devices/virtual/thermal/thermal_zone0/temp";
  int fd = open(sensorFile, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "error opening %s: %s\n", sensorFile, strerror(errno));
    return -1;
  }

  char buf[32];
  ssize_t n = read(fd, buf, 31);
  if (n < 0) {
    fprintf(stderr, "error reading %s: %s\n", sensorFile, strerror(errno));
    close(fd);
    return -1;
  }
  buf[n] = '\0';
  close(fd);
  int val = atoi(buf);
  return static_cast<float>(val) / 1000.0f; // sensor value is reported in millidegrees C
}

nvmlReturn_t checkNvmlReturn(nvmlReturn_t res, const char* op, const char* file, int line) {
  if (res != NVML_SUCCESS) {
    const char* errorDesc = nvmlErrorString(res);
    fprintf(stderr, "%s (%s:%d) returned nvmlReturn_t %d: %s\n", op, file, line, res, errorDesc);
    abort();
  }
  return res;
}

void debugPDUCommand(const char* cmd) {
  auto responseLines = pduControl.execCommand(cmd);

  printf("Exec command \"%s\" -- Response lines (%zu):\n", cmd, responseLines.size());
  for (const auto& line : responseLines) {
    printf("  \"%s\"\n", line.c_str());
  }
}

int main(int argc, char* argv[]) {
  bool useNVML = true;

  for (int i = 1; i < argc; ++i) {
    if ((!strcmp(argv[i], "--quiet")) || (!strcmp(argv[i], "-q"))) {
      quiet = true;
    } else if (!strcmp(argv[i], "--minDutyCycle")) {
      minDutyCycle = glm::clamp<float>(atof(argv[++i]), 0.0f, 1.0f);
    } else if (!strcmp(argv[i], "--maxDutyCycle")) {
      maxDutyCycle = glm::clamp<float>(atof(argv[++i]), 0.0f, 1.0f);
    } else if (!strcmp(argv[i], "--minTemp")) {
      minTemp = glm::clamp<float>(atof(argv[++i]), 0.0f, 100.0f);
    } else if (!strcmp(argv[i], "--maxTemp")) {
      maxTemp = glm::clamp<float>(atof(argv[++i]), 0.0f, 100.0f);
    } else if (!strcmp(argv[i], "--noGPU")) {
      useNVML = false;
    } else {
      printf("Unrecognized argument: %s\n", argv[i]);
      return -1;
    }
  }

  if (!pduControl.tryOpenSerial()) {
    fprintf(stderr, "error opening port\n");
    return -1;
  }

  nvmlDevice_t hDevice = 0;

  if (useNVML) {
    nvmlReturn_t initRes = nvmlInit_v2();
    if (initRes != NVML_SUCCESS) {
      printf("nvmlInit_v2() returned nvmlReturn_t %d: %s. NVML will be unavailable.\n", initRes, nvmlErrorString(initRes));
      useNVML = false;
    }
  }

  if (useNVML) {
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
    hDevice = deviceHandles[0];

    // dump temperature thresholds
    /*
    unsigned int tempShutdown, tempSlowdown, tempGpuMax; // all in degrees C
    nvmlDeviceGetTemperatureThreshold(hDevice, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN, &tempShutdown);
    nvmlDeviceGetTemperatureThreshold(hDevice, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &tempSlowdown);
    nvmlDeviceGetTemperatureThreshold(hDevice, NVML_TEMPERATURE_THRESHOLD_GPU_MAX, &tempGpuMax);
    printf("Device temperature thresholds: shutdown=%u slowdown=%u gpuMax=%u\n",
      tempShutdown, tempSlowdown, tempGpuMax);
    */
  } // useNVML

  debugPDUCommand("dmesg");

  while (true) {
    std::vector<std::string> lines;

    // Test command that should return multiple lines
    debugPDUCommand("pwr");

    // Test command with no return
    debugPDUCommand("echo 0 > /dev/led");

    float cpuCoreTemp = readThermalZoneSensor();

    unsigned int gpuCoreTemp; // in degrees C
    if (useNVML) {
      NVML_CHECK(nvmlDeviceGetTemperature(hDevice, NVML_TEMPERATURE_GPU, &gpuCoreTemp));
    } else {
      gpuCoreTemp = cpuCoreTemp;
    }

    // Fan is good all the way down to 0% duty cycle (it doesn't stop)
    float cpuDutyCycle = glm::mix(minDutyCycle, maxDutyCycle, glm::smoothstep(minTemp, maxTemp, static_cast<float>(cpuCoreTemp)));
    float gpuDutyCycle = glm::mix(minDutyCycle, maxDutyCycle, glm::smoothstep(minTemp, maxTemp, static_cast<float>(gpuCoreTemp)));

    if (!quiet) {
      printf("CPU temp %f => duty cycle %u; GPU temp %u => duty cycle %u%%\n", cpuCoreTemp, static_cast<unsigned int>(cpuDutyCycle * 100.0f), gpuCoreTemp, static_cast<unsigned int>(gpuDutyCycle * 100.0f));
    }

#if 0
    bool controlSegmentDirty = false;

    pwmSetDutyCycle(kFanCPU, cpuDutyCycle);

    if (useNVML)
      pwmSetDutyCycle(kFanGPU, gpuDutyCycle);

    if (pduInfoShm->segment()->clearRequested) {
      write(serialFd, "clear\n", 6);
      pduInfoShm->segment()->clearRequested = 0;
      controlSegmentDirty = true;
    }

    if (controlSegmentDirty) {
      pduInfoShm->flush(PDUInfo::controlSegmentStart(), PDUInfo::controlSegmentSize());
    }
#endif

    sleep(2);
  }

  nvmlShutdown();
  return 0;
}

