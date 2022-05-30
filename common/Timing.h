#pragma once
#include <stdint.h>
#include <sys/time.h>

static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

static inline float deltaTimeMs(uint64_t startTimeNs, uint64_t endTimeNs) {
  return static_cast<float>(endTimeNs - startTimeNs) / 1000000.0f;
}

