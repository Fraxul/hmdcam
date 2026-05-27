#include "rhi/vk/RHIInteropSync.h"
#include "rhi/RHI.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/vk/RHIVK.h"

RHIInteropSync::RHIInteropSync() {
  RHIVulkan* vk = rhi()->vk();
  if (!vk) {
    fprintf(stderr, "RHIInteropSync: rhi()->vk() is null; cannot allocate interop sync\n");
    abort();
  }

  // One exported timeline semaphore shared with CUDA. Counter starts at 0;
  // CUDA's first signalCUDAToRHI bumps to 1, which RHIVK's per-frame submit
  // waits on. RHIVK's per-frame submit also signals, allocating values
  // strictly greater than any CUDA-signaled value (because both sides bump
  // m_cudaSignaledValue / m_vkSignaledValue independently — the monotonicity
  // is per-side, the underlying VK counter only requires its writes to be
  // monotonic too, which holds because each side controls its own writes).
  m_timeline = vk->createExternalSemaphore(vk::SemaphoreType::eTimeline);

  // CUDA-side import. cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd
  // tells CUDA the FD points at a timeline semaphore — required for the
  // params.fence.value field on Signal/WaitExternalSemaphoresAsync to
  // mean anything.
  cudaExternalSemaphoreHandleDesc desc = {};
  desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
  desc.handle.fd = m_timeline.fd;
  CUDA_CHECK(cudaImportExternalSemaphore(&m_timelineCU, &desc));
  // cudaImportExternalSemaphore takes ownership of the FD; null out our
  // handle so the dtor doesn't double-close.
  m_timeline.fd = -1;

  // Auto-register with RHIVK so its per-frame submit can wait/signal this
  // timeline. RHIGL backend has no equivalent (and is broken-by-design for
  // live use post-Phase 6); the dynamic_cast skip leaves it untouched.
  if (RHIVK* vkRhi = dynamic_cast<RHIVK*>(rhi())) {
    vkRhi->setInteropSync(this);
  }
}

RHIInteropSync::~RHIInteropSync() {
  // Detach from RHIVK first so no submit can still reach the timeline we're
  // about to destroy.
  if (RHIVK* vkRhi = dynamic_cast<RHIVK*>(rhi())) {
    if (vkRhi->interopSync() == this) {
      vkRhi->setInteropSync(nullptr);
    }
  }
  if (m_timelineCU) {
    cudaDestroyExternalSemaphore(m_timelineCU);
    m_timelineCU = nullptr;
  }
}

void RHIInteropSync::signalCUDAToRHI(CUstream stream) {
  // Bump first so the value passed to CUDA is the one the RHI consumer
  // (RHIVK::swapBuffers) will subsequently wait on. The counter is
  // monotonically increasing — VK side reads m_cudaSignaledValue when it
  // builds the next submit's TimelineSemaphoreSubmitInfo.
  ++m_cudaSignaledValue;
  cudaExternalSemaphoreSignalParams params = {};
  params.params.fence.value = m_cudaSignaledValue;
  CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&m_timelineCU, &params, 1, stream));
}

void RHIInteropSync::signalRHIToCUDA(CUstream stream) {
  // Wait on the most recent VK-signaled value. RHIVK signals at every
  // swap/flush submit (see RHIVK::swapBuffers / RHIVK::flush), so this
  // value is whatever VK most recently committed. If no VK submit has
  // happened yet this run, m_vkSignaledValue == 0 and the wait completes
  // immediately — consistent with "nothing to wait on."
  cudaExternalSemaphoreWaitParams params = {};
  params.params.fence.value = m_vkSignaledValue;
  CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&m_timelineCU, &params, 1, stream));
}
