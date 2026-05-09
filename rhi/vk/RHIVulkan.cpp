#include "rhi/vk/RHIVulkan.h"
#include "rhi/RHI.h"
#include <epoxy/gl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE;

namespace {

// Union of extensions previously enabled by RenderBackendVKDirect, kept here
// so all VK consumers can rely on the same instance/device. The display- and
// swapchain-related entries are no-ops for backends that don't present
// through Vulkan.
const std::vector<const char*> kInstanceExtensions = {
  VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
  VK_KHR_SURFACE_EXTENSION_NAME,
  VK_KHR_DISPLAY_EXTENSION_NAME,
  VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
  VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
  VK_EXT_DISPLAY_SURFACE_COUNTER_EXTENSION_NAME,
};

const std::vector<const char*> kDeviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME,
  VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
  VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
  VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
  VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
  VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME,
};

const std::vector<const char*> kValidationLayers = {
  "VK_LAYER_LUNARG_standard_validation",
};

bool isZeroUUID(const std::array<uint8_t, VK_UUID_SIZE>& uuid) {
  for (auto b : uuid) {
    if (b)
      return false;
  }
  return true;
}

} // namespace

/*static*/ std::unique_ptr<RHIVulkan> RHIVulkan::create(const std::array<uint8_t, VK_UUID_SIZE>& gpuUUID) {
  std::unique_ptr<RHIVulkan> v(new RHIVulkan());
  try {
    PFN_vkGetInstanceProcAddr pfn = v->m_dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(pfn);

    bool enableValidation = false;
    if (const char* s = getenv("RHI_VK_ENABLE_VALIDATION")) {
      enableValidation = (atoi(s) != 0);
    }

    vk::InstanceCreateInfo ici{
      vk::InstanceCreateFlags(),
      /*applicationInfo=*/ nullptr,
      /*enabledLayerCount=*/ 0, /*ppEnabledLayerNames=*/ nullptr,
      uint32_t(kInstanceExtensions.size()), kInstanceExtensions.data()};
    if (enableValidation) {
      ici.enabledLayerCount = uint32_t(kValidationLayers.size());
      ici.ppEnabledLayerNames = kValidationLayers.data();
    }
    v->m_instance = vk::createInstanceUnique(ici);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(v->m_instance.get());

    auto devices = v->m_instance->enumeratePhysicalDevices();
    if (devices.empty()) {
      fprintf(stderr, "RHIVulkan: no physical devices found\n");
      return nullptr;
    }

    if (isZeroUUID(gpuUUID)) {
      v->m_gpu = devices[0];
    } else {
      for (const auto& d : devices) {
        auto p = d.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceIDProperties>();
        const auto& id = p.get<vk::PhysicalDeviceIDProperties>();
        if (memcmp(id.deviceUUID.data(), gpuUUID.data(), VK_UUID_SIZE) == 0) {
          v->m_gpu = d;
          break;
        }
      }
      if (!v->m_gpu) {
        fprintf(stderr, "RHIVulkan: no physical device matched the GL device UUID; falling back to device[0]\n");
        v->m_gpu = devices[0];
      }
    }

    auto props = v->m_gpu.getProperties();
    printf("RHIVulkan: selected physical device: %s\n", props.deviceName.data());

    auto families = v->m_gpu.getQueueFamilyProperties();
    bool foundQueueFamily = false;
    for (uint32_t i = 0; i < families.size(); ++i) {
      if (families[i].queueFlags & vk::QueueFlagBits::eGraphics) {
        v->m_queueFamily = i;
        foundQueueFamily = true;
        break;
      }
    }
    if (!foundQueueFamily) {
      fprintf(stderr, "RHIVulkan: no graphics queue family available\n");
      return nullptr;
    }

    float priority = 1.0f;
    vk::DeviceQueueCreateInfo qci{vk::DeviceQueueCreateFlags(), v->m_queueFamily, 1, &priority};
    vk::PhysicalDeviceFeatures features;
    vk::DeviceCreateInfo dci{
      vk::DeviceCreateFlags(),
      1, &qci,
      /*enabledLayerCount=*/ 0, /*ppEnabledLayerNames=*/ nullptr,
      uint32_t(kDeviceExtensions.size()), kDeviceExtensions.data(),
      &features};
    v->m_device = v->m_gpu.createDeviceUnique(dci);
    v->m_queue = v->m_device->getQueue(v->m_queueFamily, 0);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(v->m_device.get());

    return v;
  } catch (const std::exception& ex) {
    fprintf(stderr, "RHIVulkan::create() failed: %s\n", ex.what());
    return nullptr;
  }
}

RHIVulkan::~RHIVulkan() {
  if (m_device) {
    m_device->waitIdle();
  }
}

// Singleton ownership lives here so RHI.h can stay free of vulkan.hpp.
// Backends that allocated VK objects from this context must release them
// before this static is torn down.
static std::unique_ptr<RHIVulkan> s_rhiVk;

RHIVulkan* RHI::vk() const {
  return s_rhiVk.get();
}

void initRHIVulkan() {
  std::array<uint8_t, VK_UUID_SIZE> uuid{};
  if (epoxy_has_gl_extension("GL_EXT_memory_object")) {
    glGetUnsignedBytevEXT(GL_DEVICE_UUID_EXT, uuid.data());
  } else {
    fprintf(stderr, "initRHIVulkan: GL_EXT_memory_object not available; selecting first VK physical device without UUID match\n");
  }

  s_rhiVk = RHIVulkan::create(uuid);
  if (!s_rhiVk) {
    fprintf(stderr, "initRHIVulkan: RHIVulkan::create() failed; CUDA-GL shared surfaces will be unavailable\n");
  }
}
