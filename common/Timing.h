#pragma once
#include <stdint.h>
#include <sys/time.h>
#include <time.h>

static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

static inline float deltaTimeMs(uint64_t startTimeNs, uint64_t endTimeNs) {
  return static_cast<float>(endTimeNs - startTimeNs) / 1000000.0f;
}

static inline void delayNs(uint64_t ns) {
  struct timespec ts;
  ts.tv_sec = ns / 1000000000ULL;
  ts.tv_nsec = ns - (ts.tv_sec * 1000000000ULL);
  int res = 0;
  do {
    res = nanosleep(&ts, &ts);
  } while (res == -1 && errno == EINTR);
}

