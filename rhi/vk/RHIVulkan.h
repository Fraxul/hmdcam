#pragma once
// RHIVulkan: headless Vulkan context used as a cross-API allocator and
// synchronization-primitive factory for CUDA/GL interop. Owns instance,
// physical device, logical device, and a single queue. Does not own a surface
// or swapchain — those belong to whichever RenderBackend wants to present.

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

// vulkan.hpp triggers many -Wshadow warnings; suppress for this third-party header.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshadow"
#include "vulkan/vulkan.hpp"
#pragma clang diagnostic pop

#include <array>
#include <memory>

class RHIVulkan {
public:
  // Construct an RHIVulkan whose VkPhysicalDevice matches the given UUID.
  // Pass the all-zero UUID to skip matching and select the first enumerated
  // physical device (with a warning) — used as a fallback when the GL context
  // does not expose GL_DEVICE_UUID_EXT.
  // Returns nullptr on failure.
  static std::unique_ptr<RHIVulkan> create(const std::array<uint8_t, VK_UUID_SIZE>& gpuUUID);

  ~RHIVulkan();

  vk::Instance instance() const { return m_instance.get(); }
  vk::PhysicalDevice physicalDevice() const { return m_gpu; }
  vk::Device device() const { return m_device.get(); }
  uint32_t queueFamilyIndex() const { return m_queueFamily; }
  vk::Queue queue() const { return m_queue; }

private:
  RHIVulkan() = default;

  vk::DynamicLoader m_dl;
  vk::UniqueInstance m_instance;
  vk::PhysicalDevice m_gpu;
  uint32_t m_queueFamily = 0;
  vk::Queue m_queue;
  vk::UniqueDevice m_device;
};

// Free function: query the current GL context's GL_DEVICE_UUID_EXT (or warn
// and use the zero UUID if GL_EXT_memory_object isn't available), then
// construct the RHIVulkan singleton accessible through rhi()->vk().
// Must be called after a GL context is current and after initRHIGL().
void initRHIVulkan();
