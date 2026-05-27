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

  // Helper: pick a memory type satisfying typeFilter and propertyFlags.
  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags) const;

  // Allocate a 2D VkImage backed by exportable VkDeviceMemory. The returned
  // memoryFd is owned by the caller; dup() it for each importer (GL, CUDA).
  // Layout reports the row pitch and offset queried via
  // vkGetImageSubresourceLayout — the contractual GL/CUDA-visible layout.
  // isDedicated is set if the implementation preferred or required a
  // dedicated allocation; importers (GL, CUDA) must mark their memory
  // objects accordingly when this is true.
  struct ExternalImage {
    vk::UniqueImage image;
    vk::UniqueDeviceMemory memory;
    int memoryFd = -1;
    vk::SubresourceLayout layout;
    vk::DeviceSize allocationSize = 0;
    bool isDedicated = false;
  };
  ExternalImage allocateExternalImage(
    uint32_t width, uint32_t height,
    vk::Format format, vk::ImageUsageFlags usage,
    vk::ImageTiling tiling = vk::ImageTiling::eLinear) const;

  // Allocate a VkBuffer backed by exportable VkDeviceMemory. As with images,
  // memoryFd ownership transfers to the caller.
  struct ExternalBuffer {
    vk::UniqueBuffer buffer;
    vk::UniqueDeviceMemory memory;
    int memoryFd = -1;
    vk::DeviceSize allocationSize = 0;
  };
  ExternalBuffer allocateExternalBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) const;

  // Allocate a VkSemaphore backed by an exportable POSIX FD handle. The
  // returned fd is owned by the caller; dup() for each importer. Pass
  // vk::SemaphoreType::eTimeline to create a timeline semaphore (initial
  // counter value 0) instead of the default binary semaphore.
  struct ExternalSemaphore {
    vk::UniqueSemaphore semaphore;
    int fd = -1;
  };
  ExternalSemaphore createExternalSemaphore(vk::SemaphoreType = vk::SemaphoreType::eBinary) const;

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
//
// This is the "VK allocator attached to a GL backend" path used for
// CUDA/GL interop. For the top-level Vulkan RHI backend, see
// initRHIVulkan() in rhi/vk/RHIVK.cpp.
void initRHIVulkanInteropContext();

// Install the headless RHIVulkan singleton accessible via rhi()->vk(). Used by
// both initRHIVulkanInteropContext (GL backend's interop helper) and the
// top-level initRHIVulkan (RHIVK backend) so the singleton-management code
// lives in one place. Pass the all-zero UUID to select the first enumerated
// physical device. Returns true on success; asserts if a singleton is already
// installed.
bool rhiVulkanInstallSingleton(const std::array<uint8_t, VK_UUID_SIZE>& gpuUUID);
