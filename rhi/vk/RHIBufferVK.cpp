#include "rhi/vk/RHIBufferVK.h"
#include "rhi/RHI.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

RHIBufferVK::RHIBufferVK() {
  m_data = nullptr;
  m_size = 0;
  m_usageMode = kBufferUsageCPUWriteOnly;
}

RHIBufferVK::~RHIBufferVK() {
  // m_buffer / m_memory are UniqueXxx — auto-destroyed.
}

/*static*/ RHIBufferVK::ptr RHIBufferVK::create(size_t size, RHIBufferUsageMode mode, const void* initialContents) {
  if (size == 0) {
    fprintf(stderr, "RHIBufferVK::create: zero-size buffer requested\n");
    abort();
  }
  vk::Device device = rhi()->vk()->device();

  // Broad usage flags: the RHI doesn't distinguish vertex vs index vs uniform
  // vs storage etc. at allocation time, so we permit anything. The driver
  // doesn't actually care for host-visible buffers — it's just a hint.
  vk::BufferUsageFlags usageFlags =
    vk::BufferUsageFlagBits::eTransferSrc |
    vk::BufferUsageFlagBits::eTransferDst |
    vk::BufferUsageFlagBits::eVertexBuffer |
    vk::BufferUsageFlagBits::eIndexBuffer |
    vk::BufferUsageFlagBits::eUniformBuffer |
    vk::BufferUsageFlagBits::eStorageBuffer |
    vk::BufferUsageFlagBits::eIndirectBuffer;

  RHIBufferVK::ptr buf(new RHIBufferVK());
  buf->m_size = size;
  buf->m_usageMode = mode;

  vk::BufferCreateInfo bufCi{vk::BufferCreateFlags(), size, usageFlags, vk::SharingMode::eExclusive};
  buf->m_buffer = device.createBufferUnique(bufCi);

  vk::MemoryRequirements memReq = device.getBufferMemoryRequirements(buf->m_buffer.get());
  uint32_t memTypeIndex = rhi()->vk()->findMemoryType(
    memReq.memoryTypeBits,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

  vk::MemoryAllocateInfo allocInfo{memReq.size, memTypeIndex};
  buf->m_memory = device.allocateMemoryUnique(allocInfo);
  device.bindBufferMemory(buf->m_buffer.get(), buf->m_memory.get(), 0);

  if (initialContents) {
    buf->loadData(initialContents, size, 0);
  }
  return buf;
}

void RHIBufferVK::loadData(const void* src, size_t length, size_t offset) {
  if (offset + length > m_size) {
    fprintf(stderr, "RHIBufferVK::loadData: write past end of buffer (offset=%zu length=%zu size=%zu)\n",
      offset, length, m_size);
    abort();
  }
  vk::Device device = rhi()->vk()->device();
  void* mapped = device.mapMemory(m_memory.get(), offset, length);
  memcpy(mapped, src, length);
  device.unmapMemory(m_memory.get());
}

void RHIBufferVK::map(RHIBufferMapMode) {
  // Map the whole buffer; expose via m_data.
  // Coherent memory means no explicit flushes are needed.
  m_data = rhi()->vk()->device().mapMemory(m_memory.get(), 0, VK_WHOLE_SIZE);
}

void RHIBufferVK::unmap() {
  rhi()->vk()->device().unmapMemory(m_memory.get());
  m_data = nullptr;
}
