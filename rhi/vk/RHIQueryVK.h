#pragma once
// RHIQueryVK: timer and occlusion query objects for the VK backend.
//
// Each query owns its own small VkQueryPool. Per-pool overhead is tiny and
// keeping the pool 1:1 with the RHI object means lifetime is trivial — no
// shared pool to garbage-collect, no slot recycling. Engine creates a
// handful of timer queries at process start and reuses them across frames;
// host-side reset (VK 1.2 core, enabled in RHIVulkan) lets begin/record run
// without an in-CB vkCmdResetQueryPool.
//
// Timer query semantics match the GL backend: bracket with
// begin/endTimerQuery and read getQueryResult on a later frame for elapsed
// nanoseconds. recordTimestamp records a single absolute timestamp.
//
// All begin/end/recordTimestamp calls record into RHIVK's per-frame
// command buffer (see RHIVK::ensureFrameCBOpen). Calling them outside
// any begin*..swapBuffers scope still works — the helper lazily opens
// the frame CB on first use — so the engine's habit of calling
// beginTimerQuery before the first beginRenderPass is fine.

#include "rhi/RHIQuery.h"
#include "rhi/vk/RHIVulkan.h"

class RHITimerQueryVK : public RHITimerQuery {
public:
  typedef boost::intrusive_ptr<RHITimerQueryVK> ptr;
  RHITimerQueryVK();
  virtual ~RHITimerQueryVK();

  // Two-slot pool: index 0 = begin / single timestamp, index 1 = end.
  vk::QueryPool queryPool() const { return m_queryPool.get(); }

  // State the query is in. Drives validation in
  // begin/end/recordTimestamp and getQueryResult.
  enum class Kind {
    kEmpty, // never written or already read
    kElapsedBegun, // begin written, awaiting end
    kElapsedEnded, // both begin + end written, awaiting readback
    kTimestamp, // single timestamp written, awaiting readback
  };
  Kind kind() const { return m_kind; }
  void setKind(Kind k) { m_kind = k; }

protected:
  vk::UniqueQueryPool m_queryPool;
  Kind m_kind = Kind::kEmpty;
};

class RHIOcclusionQueryVK : public RHIOcclusionQuery {
public:
  typedef boost::intrusive_ptr<RHIOcclusionQueryVK> ptr;
  RHIOcclusionQueryVK(RHIOcclusionQueryMode);
  virtual ~RHIOcclusionQueryVK();

  vk::QueryPool queryPool() const { return m_queryPool.get(); }
  RHIOcclusionQueryMode mode() const { return m_mode; }

protected:
  vk::UniqueQueryPool m_queryPool;
  RHIOcclusionQueryMode m_mode;
};
