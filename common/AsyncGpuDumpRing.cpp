#include "common/AsyncGpuDumpRing.h"
#include "common/FxThreading.h"
#include "common/Timing.h"
#include "rhi/cuda/RHICUDA.h"
#include <utility>

AsyncGpuDumpRing::AsyncGpuDumpRing(size_t slotCount, size_t slotBytes) :
  m_slotCount(slotCount) {
  m_slots = new Slot[m_slotCount];

  std::scoped_lock l(m_freeListLock);
  for (size_t i = 0; i < m_slotCount; ++i) {
    Slot& slot = m_slots[i];
    slot.capacity = slotBytes;
    CUDA_CHECK(cuMemAllocHost(&slot.hostPtr, slotBytes));
    CUDA_CHECK(cuEventCreate(&slot.copyDoneEvent, CU_EVENT_BLOCKING_SYNC | CU_EVENT_DISABLE_TIMING));
    m_freeList.push(&slot);
  }
}

AsyncGpuDumpRing::~AsyncGpuDumpRing() {
  // Workers hold a raw `this` pointer, so we must not free anything until they have all returned.
  drainBlocking();

  for (size_t i = 0; i < m_slotCount; ++i) {
    cuMemFreeHost(m_slots[i].hostPtr);
    cuEventDestroy(m_slots[i].copyDoneEvent);
  }
  delete[] m_slots;
}

AsyncGpuDumpRing::Slot* AsyncGpuDumpRing::acquire() {
  std::scoped_lock l(m_freeListLock);
  if (m_freeList.empty())
    return nullptr; // Pool exhausted -- caller treats this as a dropped frame.

  Slot* slot = m_freeList.front();
  m_freeList.pop();
  return slot;
}

void AsyncGpuDumpRing::dispatch(Slot* slot, std::function<void(Slot*)> writeFn) {
  FxThreading::runFunction([this, slot, writeFn = std::move(writeFn)]() {
    // Wait for the caller's device->host copy to complete, serialize the buffer, then recycle.
    CUDA_CHECK(cuEventSynchronize(slot->copyDoneEvent));
    writeFn(slot);

    std::scoped_lock l(m_freeListLock);
    m_freeList.push(slot);
  });
}

size_t AsyncGpuDumpRing::outstanding() const {
  std::scoped_lock l(m_freeListLock);
  return m_slotCount - m_freeList.size();
}

void AsyncGpuDumpRing::drainBlocking() {
  while (outstanding() != 0)
    delayMs(1); // Yield the CPU while in-flight writes finish.
}
