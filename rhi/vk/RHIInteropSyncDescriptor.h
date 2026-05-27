#pragma once
#include "rhi/vk/RHIInteropSync.h"

// Lightweight descriptor passed to interop buffer/surface factories so the
// allocated resource can find its associated RHIInteropSync. The timeline
// semaphore is shared globally across all resources registered with the
// same RHIInteropSync — there is no per-resource direction or barrier
// metadata; that's all expressed through the timeline counter at submit
// time.
class RHIInteropSyncDescriptor {
public:
  RHIInteropSyncDescriptor() = default;
  RHIInteropSyncDescriptor(RHIInteropSync::ptr sync) :
    m_sync(sync) {}

  RHIInteropSync::ptr sync() const { return m_sync; }

protected:
  RHIInteropSync::ptr m_sync;
};
