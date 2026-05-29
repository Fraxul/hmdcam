#pragma once
// RHIBufferVK: Vulkan buffer.
//
// Currently, all buffers use host-visible + host-coherent memory so we don't need
// to manage staging buffers. Fine for Tegra UMA where there's minimal performance
// difference, but this may have to change on other hardware.

#include "rhi/RHIBuffer.h"
#include "rhi/vk/RHIVulkan.h"

class RHIBufferVK : public RHIBuffer {
public:
  typedef boost::intrusive_ptr<RHIBufferVK> ptr;

  // Allocate a buffer of `size` bytes. If `initialContents` is non-null,
  // initialize the buffer with that data.
  static RHIBufferVK::ptr create(size_t size, RHIBufferUsageMode mode, const void* initialContents);

  virtual ~RHIBufferVK();

  virtual void map(RHIBufferMapMode) override;
  virtual void unmap() override;

  vk::Buffer vkBuffer() const { return m_buffer.get(); }

  // Upload data to a region of the buffer.
  void loadData(const void* src, size_t length, size_t offset);

protected:
  RHIBufferVK();

  vk::UniqueBuffer m_buffer;
  vk::UniqueDeviceMemory m_memory;
};
