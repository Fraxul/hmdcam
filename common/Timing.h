#pragma once
#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>

#ifdef IS_TEGRA
// Returns monotonic time (time since boot)
// Uses the TSC on Tegra platforms, which is fixed at 31.250MHz.
// libargus hardware timestamps use the TSC, so this makes timebase conversion trivial.
static inline uint64_t tscTimestampToNs(uint64_t tscTimestamp) {
  // 31.250MHz == 32 ns/tick.
  return tscTimestamp * 32u;
}

static inline uint64_t currentTimeNs() {
  uint64_t tscCounter;
  asm volatile("mrs %0, cntvct_el0" : "=r"(tscCounter));
  return tscTimestampToNs(tscCounter);
}
#else
// Returns monotonic time (time since boot).
// Uses CLOCK_MONOTONIC_RAW to avoid the timebase drifting during clock adjustment.
// Be careful if interacting with events timestamped by other parts of the system that may use CLOCK_MONOTONIC.
static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}
#endif

// Returns realtime (time since the 1970 epoch).
// Uses the CLOCK_REALTIME_COARSE timer, which is accurate to about a millisecond.
static inline uint64_t currentRealTimeMs() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME_COARSE, &ts);
  return (ts.tv_sec * 1000ULL) + (ts.tv_nsec / 1000000);
}

static inline float deltaTimeMs(uint64_t startTimeNs, uint64_t endTimeNs) {
  return static_cast<float>(static_cast<int64_t>(endTimeNs) - static_cast<int64_t>(startTimeNs)) / 1000000.0f;
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

class PerfTimer {
public:
  PerfTimer() {
    startTimeNs = currentTimeNs();
    lastCheckpointTimeNs = startTimeNs;
  }

  // Returns time elapsed since previous checkpoint (or construction) in milliseconds
  float checkpoint() {
    uint64_t now = currentTimeNs();
    float res = deltaTimeMs(lastCheckpointTimeNs, now);
    lastCheckpointTimeNs = now;
    return res;
  }

  // Returns time elapsed since construction in milliseconds. Does not update the checkpoint timer.
  float totalElapsedTime() {
    return deltaTimeMs(startTimeNs, currentTimeNs());
  }

  uint64_t startTimeNs;
  uint64_t lastCheckpointTimeNs;
};

