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
#include "SHMSegment.h"
#include "PDUSHM.h"

#include <termios.h>
#include <unistd.h>

#include <vector>
#include <string>

#define NVML_CHECK(x) checkNvmlReturn(x, #x, __FILE__, __LINE__)

int serialFd;
bool quiet = false;
SHMSegment<PDUInfo>* pduInfoShm;
float minDutyCycle = 0.0f;
float maxDutyCycle = 1.0f;
float minTemp = 30.0f;
float maxTemp = 60.0f;

enum FanID {
  kFanCPU,
  kFanGPU
};

static inline uint64_t currentTimeMs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000ULL) + (ts.tv_nsec / 1000000ULL);
}


// Bus 20.03V 0.04A 0.87W UsedPower 3228.17J 0.90Wh PD 20.00V 5.00A
static boost::regex infoExpr("Bus ([\\d\\.]+)V ([\\d\\.]+)A ([\\d\\.]+)W UsedPower ([\\d\\.]+)J ([\\d\\.]+)Wh");

// 0-1 range
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

void drainSerialInput() {
  struct pollfd pfd;
  pfd.fd = serialFd;
  pfd.events = POLLIN;
  pfd.revents = 0;

  while (true) {
    int pollRes = poll(&pfd, 1, 0);
    if (!pollRes)
      break;

    // Extra data in the serial input buffer, pass it through
    char buf[1024];
    ssize_t n = read(serialFd, buf, 1023);
    buf[n] = '\0';


    const char *start = buf, *end = buf + n;
    boost::match_results<const char*> what;
    boost::match_flag_type flags = boost::match_default;
    while(boost::regex_search(start, end, what, infoExpr, flags)) {
      try {
        // what[0] contains the whole string, capture groups start at what[1]
        pduInfoShm->segment()->busVoltage  = boost::lexical_cast<float>(std::string(what[1].first, what[1].second));
        pduInfoShm->segment()->busAmperage = boost::lexical_cast<float>(std::string(what[2].first, what[2].second));
        pduInfoShm->segment()->busPower    = boost::lexical_cast<float>(std::string(what[3].first, what[3].second));
        pduInfoShm->segment()->usedPowerJ  = boost::lexical_cast<float>(std::string(what[4].first, what[4].second));
        pduInfoShm->segment()->usedPowerWH = boost::lexical_cast<float>(std::string(what[5].first, what[5].second));
        pduInfoShm->segment()->lastUpdateTimeMs = currentTimeMs();
        pduInfoShm->flush(PDUInfo::dataSegmentStart(), PDUInfo::dataSegmentSize());
        //printf("Match: %f %f %f %f %f\n",
        //  info.busVoltage, info.busAmperage, info.busPower, info.usedPowerJ, info.usedPowerWH);

        // update search position:
        start = what[0].second;
        // update flags:
        flags |= boost::match_prev_avail;
        flags |= boost::match_not_bob;
      } catch (const std::exception& ex) {
        fprintf(stderr, "Parse error: %s\nInput was: \"\"\"%s\"\"\"\n\n", ex.what(), start);
      }
    }

    if (!quiet)
      printf("%s", buf);
  }
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
  std::string serialPort = "/dev/ttyTHS0";
  bool useNVML = true;

  for (int i = 1; i < argc; ++i) {
    if ((!strcmp(argv[i], "--quiet")) || (!strcmp(argv[i], "-q"))) {
      quiet = true;
    } else if (!strcmp(argv[i], "--port")) {
      serialPort = argv[++i];
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

  pduInfoShm = SHMSegment<PDUInfo>::createSegment("pdu-info", sizeof(PDUInfo));
  if (!pduInfoShm) {
    fprintf(stderr, "can't create/open pdu-info shm segment\n");
    return -1;
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

  while (true) {
    drainSerialInput();

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

    pwmSetDutyCycle(kFanCPU, cpuDutyCycle);

    if (useNVML)
      pwmSetDutyCycle(kFanGPU, gpuDutyCycle);

    bool controlSegmentDirty = false;
    if (pduInfoShm->segment()->clearRequested) {
      write(serialFd, "clear\n", 6);
      pduInfoShm->segment()->clearRequested = 0;
      controlSegmentDirty = true;
    }

    if (controlSegmentDirty) {
      pduInfoShm->flush(PDUInfo::controlSegmentStart(), PDUInfo::controlSegmentSize());
    }

    // request info which will be read next cycle
    write(serialFd, "info\n", 5);

    sleep(2);
  }

  nvmlShutdown();
  return 0;
}

