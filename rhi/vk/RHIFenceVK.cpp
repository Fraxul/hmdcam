#include "rhi/vk/RHIFenceVK.h"
#include "rhi/RHI.h"
#include <string>

RHIFenceVK::RHIFenceVK(vk::Semaphore timeline, uint64_t targetValue) :
  m_timeline(timeline),
  m_targetValue(targetValue) {}

RHIFenceVK::~RHIFenceVK() {}

bool RHIFenceVK::wait(uint64_t timeoutNs) {
  vk::Device device = rhi()->vk()->device();
  vk::SemaphoreWaitInfo waitInfo{};
  waitInfo.setSemaphores(m_timeline);
  waitInfo.setValues(m_targetValue);
  vk::Result res = device.waitSemaphores(waitInfo, timeoutNs);
  if (res == vk::Result::eSuccess)
    return true;
  else if (res == vk::Result::eTimeout)
    return false;
  // Other failure -- log and return false
  const std::string& errStr = vk::to_string(res);
  fprintf(stderr, "RHIFenceVK(%p)::wait(value=%lu, %lu ns): %s\n", this, m_targetValue, timeoutNs, errStr.c_str());
  return false;
}

bool RHIFenceVK::isSignaled() {
  vk::Device device = rhi()->vk()->device();
  return device.getSemaphoreCounterValue(m_timeline) >= m_targetValue;
}
