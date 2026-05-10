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
  VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
  VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
  VK_EXT_DISPLAY_CONTROL_EXTENSION_NAME,
};

const std::vector<const char*> kValidationLayers = {
  "VK_LAYER_KHRONOS_validation",
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

    // When validation is enabled, also turn on synchronization validation —
    // it's the layer that catches binary-semaphore signal/wait misuse, and
    // it isn't on by default in the Khronos validation layer.
    std::vector<const char*> instanceExtensions(kInstanceExtensions.begin(), kInstanceExtensions.end());
    if (enableValidation) {
      instanceExtensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
    }
    static const vk::ValidationFeatureEnableEXT kEnabledValidationFeatures[] = {
      vk::ValidationFeatureEnableEXT::eSynchronizationValidation,
    };
    vk::ValidationFeaturesEXT validationFeatures{
      uint32_t(sizeof(kEnabledValidationFeatures) / sizeof(kEnabledValidationFeatures[0])), kEnabledValidationFeatures,
      0, nullptr};

    vk::InstanceCreateInfo ici{
      vk::InstanceCreateFlags(),
      /*applicationInfo=*/ nullptr,
      /*enabledLayerCount=*/ 0, /*ppEnabledLayerNames=*/ nullptr,
      uint32_t(instanceExtensions.size()), instanceExtensions.data()};
    if (enableValidation) {
      ici.enabledLayerCount = uint32_t(kValidationLayers.size());
      ici.ppEnabledLayerNames = kValidationLayers.data();
      ici.setPNext(&validationFeatures);
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

uint32_t RHIVulkan::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const {
  vk::PhysicalDeviceMemoryProperties memProperties = m_gpu.getMemoryProperties();
  for (uint32_t i = 0; i < memProperties.memoryTypeCount; ++i) {
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
      return i;
    }
  }
  fprintf(stderr, "RHIVulkan::findMemoryType: no memory type satisfies typeFilter=0x%08x properties=0x%08x\n",
    typeFilter, static_cast<uint32_t>(properties));
  abort();
}

RHIVulkan::ExternalImage RHIVulkan::allocateExternalImage(
  uint32_t width, uint32_t height,
  vk::Format format, vk::ImageUsageFlags usage, vk::ImageTiling tiling) const {

  ExternalImage out;

  // clang-format off
  vk::StructureChain<vk::ImageCreateInfo, vk::ExternalMemoryImageCreateInfo> ici = {
    vk::ImageCreateInfo{
      vk::ImageCreateFlags(),
      vk::ImageType::e2D,
      format,
      vk::Extent3D(width, height, 1),
      /*mipLevels=*/ 1,
      /*arrayLayers=*/ 1,
      vk::SampleCountFlagBits::e1,
      tiling,
      usage,
      vk::SharingMode::eExclusive,
      /*queueFamilies=*/ 0, nullptr,
      /*initialLayout=*/ vk::ImageLayout::eUndefined
    },
    vk::ExternalMemoryImageCreateInfo{
      vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd
    }
  };
  // clang-format on

  out.image = m_device->createImageUnique(ici.get());

  // Query memory requirements with a MemoryDedicatedRequirements chain so we
  // can detect cases (common on NVIDIA, especially for linear-tiled external
  // images) where the implementation needs a dedicated allocation.
  auto memReqChain = m_device->getImageMemoryRequirements2<vk::MemoryRequirements2, vk::MemoryDedicatedRequirements>(
    {out.image.get()});
  const vk::MemoryRequirements& memReq = memReqChain.get<vk::MemoryRequirements2>().memoryRequirements;
  const vk::MemoryDedicatedRequirements& dedReq = memReqChain.get<vk::MemoryDedicatedRequirements>();
  out.isDedicated = (dedReq.requiresDedicatedAllocation || dedReq.prefersDedicatedAllocation);

  // One-shot log so the first interop allocation tells you what the driver
  // wants. Useful for tracking down INVALID_VALUE on glTexStorageMem2DEXT.
  static bool s_loggedDedicated = false;
  if (!s_loggedDedicated) {
    s_loggedDedicated = true;
    fprintf(stderr, "RHIVulkan::allocateExternalImage: prefersDedicated=%u requiresDedicated=%u (using dedicated=%u)\n",
      dedReq.prefersDedicatedAllocation, dedReq.requiresDedicatedAllocation, out.isDedicated);
  }

  vk::MemoryAllocateInfo mai{memReq.size, findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlags())};
  vk::ExportMemoryAllocateInfo emai(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
  vk::MemoryDedicatedAllocateInfo mdai(out.image.get(), /*buffer=*/ VK_NULL_HANDLE);

  // pNext chain: MemoryAllocateInfo -> ExportMemoryAllocateInfo -> [optional] MemoryDedicatedAllocateInfo.
  mai.setPNext(&emai);
  if (out.isDedicated)
    emai.setPNext(&mdai);

  out.memory = m_device->allocateMemoryUnique(mai);
  m_device->bindImageMemory(out.image.get(), out.memory.get(), 0);

  vk::MemoryGetFdInfoKHR getFd{out.memory.get(), vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd};
  out.memoryFd = m_device->getMemoryFdKHR(getFd);

  // Subresource layout is only meaningful for linear-tiled images; for optimal
  // tiling the values are unspecified. Caller is responsible for not relying
  // on the pitch in that case.
  if (tiling == vk::ImageTiling::eLinear) {
    vk::ImageSubresource subres{vk::ImageAspectFlagBits::eColor, 0, 0};
    out.layout = m_device->getImageSubresourceLayout(out.image.get(), subres);
  } else {
    out.layout = vk::SubresourceLayout{};
  }
  out.allocationSize = memReq.size;

  return out;
}

RHIVulkan::ExternalBuffer RHIVulkan::allocateExternalBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage) const {
  ExternalBuffer out;

  // clang-format off
  vk::StructureChain<vk::BufferCreateInfo, vk::ExternalMemoryBufferCreateInfo> bci = {
    vk::BufferCreateInfo{
      vk::BufferCreateFlags(),
      size,
      usage,
      vk::SharingMode::eExclusive
    },
    vk::ExternalMemoryBufferCreateInfo{
      vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd
    }
  };
  // clang-format on

  out.buffer = m_device->createBufferUnique(bci.get());

  vk::MemoryRequirements memReq = m_device->getBufferMemoryRequirements(out.buffer.get());
  vk::MemoryAllocateInfo mai{memReq.size, findMemoryType(memReq.memoryTypeBits, vk::MemoryPropertyFlags())};
  vk::ExportMemoryAllocateInfo emai(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
  mai.setPNext(&emai);

  out.memory = m_device->allocateMemoryUnique(mai);
  m_device->bindBufferMemory(out.buffer.get(), out.memory.get(), 0);

  vk::MemoryGetFdInfoKHR getFd{out.memory.get(), vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd};
  out.memoryFd = m_device->getMemoryFdKHR(getFd);
  out.allocationSize = memReq.size;

  return out;
}

RHIVulkan::ExternalSemaphore RHIVulkan::createExternalSemaphore() const {
  ExternalSemaphore out;

  vk::SemaphoreCreateInfo sci{};
  vk::ExportSemaphoreCreateInfo esci{vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd};
  sci.setPNext(&esci);

  out.semaphore = m_device->createSemaphoreUnique(sci);

  vk::SemaphoreGetFdInfoKHR getFd{out.semaphore.get(), vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd};
  out.fd = m_device->getSemaphoreFdKHR(getFd);

  return out;
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
