#pragma once
#include <stdint.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>

// Returns monotonic time (time since boot)
// This really should be CLOCK_MONOTONIC_RAW to avoid the timebase drifting during clock adjustment,
// but many events we care about (like libargus capture timestamps, or DRM page-flip timestamps) are
// relative to CLOCK_MONOTONIC, so using that is the path of least resistance.
static inline uint64_t currentTimeNs() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (ts.tv_sec * 1000000000ULL) + ts.tv_nsec;
}

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

