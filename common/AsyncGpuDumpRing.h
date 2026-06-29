#pragma once
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <cuda.h>

// A fixed-size pool of pinned host buffers, each paired with a CUDA completion event, plus a
// worker-thread dispatch path. It exists to stream GPU surfaces to disk with minimal CPU
// overhead: the device->host copy runs async on the GPU, and the (potentially blocking) file
// write happens on a worker thread once the copy completes.
//
// Lifecycle of one capture:
//   1. acquire() a Slot. Returns nullptr if the pool is currently exhausted -- the caller treats
//      that as a dropped frame.
//   2. Issue one or more async device->host copies into slot->hostPtr, then record
//      slot->copyDoneEvent on the same stream.
//   3. dispatch() the slot with a write callback. The ring waits on the event on a worker thread,
//      invokes the callback to serialize the buffer, then returns the slot to the pool.
//
// The ring owns the pinned buffers and events and manages slot recycling. It knows nothing about
// file formats, naming, or metadata -- those are the caller's policy, expressed in the callback.
class AsyncGpuDumpRing {
public:
  struct Slot {
    void* hostPtr = nullptr; // Pinned host buffer, `capacity` bytes.
    CUevent copyDoneEvent = 0; // Caller records this after issuing copies into hostPtr.
    size_t capacity = 0;
  };

  // Allocates slotCount pinned buffers of slotBytes each.
  AsyncGpuDumpRing(size_t slotCount, size_t slotBytes);
  ~AsyncGpuDumpRing();

  AsyncGpuDumpRing(const AsyncGpuDumpRing&) = delete;
  AsyncGpuDumpRing& operator=(const AsyncGpuDumpRing&) = delete;

  // Pops a free slot, or returns nullptr if the pool is currently exhausted.
  Slot* acquire();

  // Queues async work: wait on slot->copyDoneEvent, invoke writeFn(slot) to serialize the buffer,
  // then return the slot to the pool. Non-blocking; writeFn runs on a worker thread.
  void dispatch(Slot*, std::function<void(Slot*)> writeFn);

  // Number of slots currently checked out (acquired but not yet returned by their worker).
  size_t outstanding() const;

  // Blocks until every outstanding slot has been returned. Intended for shutdown paths, where the
  // caller must know all in-flight writes have finished before tearing down their output state.
  void drainBlocking();

  size_t slotCount() const { return m_slotCount; }

private:
  size_t m_slotCount;
  Slot* m_slots = nullptr;
  std::queue<Slot*> m_freeList;
  mutable std::mutex m_freeListLock;
};
