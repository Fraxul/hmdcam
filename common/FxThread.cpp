#include "common/FxThread.h"
#include <sched.h>

namespace {

cpu_set_t g_defaultSet;
bool g_defaultValid = false;

// Idempotent: captures the calling process's affinity mask the first time it establishes a
// valid default. Safe to call from any thread once the static-init capture below has run
// (which happens before main(), hence before any thread is spawned).
void captureProcessDefaultOnce() {
  if (g_defaultValid)
    return;
  CPU_ZERO(&g_defaultSet);
  if (sched_getaffinity(0, sizeof g_defaultSet, &g_defaultSet) == 0 && CPU_COUNT(&g_defaultSet) > 0)
    g_defaultValid = true;
}

// Capture at static-initialization time.
// With isolcpus, the inherited process affinity already excludes the isolated core(s), so this
// records exactly the housekeeping set that worker threads should inherit.
const bool g_defaultInit = (captureProcessDefaultOnce(), true);

} // namespace

bool FxThread::getDefaultAffinity(cpu_set_t& outSet) {
  captureProcessDefaultOnce();
  CPU_ZERO(&outSet);
  if (!g_defaultValid)
    return false;
  outSet = g_defaultSet;
  return true;
}

void FxThread::setDefaultAffinity(const cpu_set_t& cpuSet) {
  g_defaultSet = cpuSet;
  g_defaultValid = (CPU_COUNT(&g_defaultSet) > 0);
}
