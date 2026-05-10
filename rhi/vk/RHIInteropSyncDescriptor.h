#pragma once
#include "rhi/vk/RHIInteropSync.h"

enum RHIInteropSyncDirection {
  kSyncDirectionNone = 0,
  kSyncDirectionCUDAWriter = 0x1,
  kSyncDirectionRHIWriter = 0x2,
  kSyncDirectionBidirectional = 0x3, // both APIs write
};

class RHIInteropSyncDescriptor {
public:
  RHIInteropSyncDescriptor() :
    m_direction(kSyncDirectionNone) {}

  RHIInteropSyncDescriptor(RHIInteropSync::ptr sync, RHIInteropSyncDirection direction) :
    m_sync(sync),
    m_direction(direction) {}

  RHIInteropSync::ptr sync() const { return m_sync; }
  RHIInteropSyncDirection direction() const { return m_direction; }

protected:
  RHIInteropSync::ptr m_sync;
  RHIInteropSyncDirection m_direction;
};
