#include "rhi/vk/RHIQueryVK.h"
#include "rhi/RHI.h"

RHITimerQueryVK::RHITimerQueryVK() {
  vk::Device device = rhi()->vk()->device();
  vk::QueryPoolCreateInfo ci{vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, /*queryCount=*/ 2};
  m_queryPool = device.createQueryPoolUnique(ci);
  // Reset the pool so the first begin/record can write without an
  // additional reset call (the slots are in "unavailable" state at
  // creation regardless of pool type, and writeTimestamp expects them to
  // be reset first).
  device.resetQueryPool(m_queryPool.get(), 0, 2);
}

RHITimerQueryVK::~RHITimerQueryVK() {}

RHIOcclusionQueryVK::RHIOcclusionQueryVK(RHIOcclusionQueryMode mode) :
  m_mode(mode) {
  vk::Device device = rhi()->vk()->device();
  vk::QueryPoolCreateInfo ci{vk::QueryPoolCreateFlags(), vk::QueryType::eOcclusion, /*queryCount=*/ 1};
  m_queryPool = device.createQueryPoolUnique(ci);
  device.resetQueryPool(m_queryPool.get(), 0, 1);
}

RHIOcclusionQueryVK::~RHIOcclusionQueryVK() {}
