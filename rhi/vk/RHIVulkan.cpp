#include "rhi/vk/RHIVulkan.h"
#include "rhi/RHI.h"
#include <epoxy/gl.h>
#include <assert.h>
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
  // VK-native RHI backend needs dynamic rendering (replaces VkRenderPass /
  // VkFramebuffer ceremony with vkCmdBeginRendering / vkCmdEndRendering).
  VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
  // Shader objects + the dynamic state extensions they pull in. With
  // shader_object enabled we never construct a VkPipeline for graphics — all
  // rasterizer/blend/depth/vertex-input state is set per-draw via
  // vkCmdSet*EXT. Together this matches the GL RHI's "rebind anything
  // anywhere" model far more naturally than monolithic VkPipeline objects.
  VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
  VK_EXT_EXTENDED_DYNAMIC_STATE_EXTENSION_NAME,
  VK_EXT_EXTENDED_DYNAMIC_STATE_2_EXTENSION_NAME,
  VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
  VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME,
  // Push descriptors let us issue resource bindings without managing a
  // descriptor pool / per-frame allocation — vkCmdPushDescriptorSetKHR
  // pushes a transient set inline into the command buffer.
  VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
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

    // apiVersion controls which core API the loader/validation expect us
    // to be using; 1.3 covers dynamic rendering, shader_object's
    // dependency chain, and the extended-dynamic-state extensions.
    vk::ApplicationInfo appInfo{"hmdcam", 1, "hmdcam", 1, VK_API_VERSION_1_3};

    vk::InstanceCreateInfo ici{
      vk::InstanceCreateFlags(),
      &appInfo,
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

    // Enable a minimal set of "core" features that several pipelines and
    // shaders the engine compiles end up using. Adding to this when
    // validation reports a missing feature is cheaper than chasing
    // each one individually.
    vk::PhysicalDeviceFeatures features{};
    features.geometryShader = VK_TRUE;
    features.fillModeNonSolid = VK_TRUE;
    features.wideLines = VK_TRUE;
    features.samplerAnisotropy = VK_TRUE;
    features.fragmentStoresAndAtomics = VK_TRUE;
    features.vertexPipelineStoresAndAtomics = VK_TRUE;
    features.shaderStorageImageExtendedFormats = VK_TRUE;
    features.multiViewport = VK_TRUE;

    // Feature chain for dynamic rendering + shader_object + the dynamic state
    // extensions shader_object depends on. vulkan-hpp's StructureChain
    // would be cleaner but bringing the structs in by hand keeps the
    // dependency on optional features visible and lets us toggle each as
    // their presence is validated.
    vk::PhysicalDeviceDynamicRenderingFeaturesKHR dynamicRenderingFeatures{};
    dynamicRenderingFeatures.dynamicRendering = VK_TRUE;

    vk::PhysicalDeviceShaderObjectFeaturesEXT shaderObjectFeatures{};
    shaderObjectFeatures.shaderObject = VK_TRUE;

    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT eds1Features{};
    eds1Features.extendedDynamicState = VK_TRUE;

    vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT eds2Features{};
    eds2Features.extendedDynamicState2 = VK_TRUE;
    eds2Features.extendedDynamicState2LogicOp = VK_TRUE;

    vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT eds3Features{};
    eds3Features.extendedDynamicState3DepthClampEnable = VK_TRUE;
    eds3Features.extendedDynamicState3PolygonMode = VK_TRUE;
    eds3Features.extendedDynamicState3RasterizationSamples = VK_TRUE;
    eds3Features.extendedDynamicState3SampleMask = VK_TRUE;
    eds3Features.extendedDynamicState3AlphaToCoverageEnable = VK_TRUE;
    eds3Features.extendedDynamicState3AlphaToOneEnable = VK_TRUE;
    eds3Features.extendedDynamicState3LogicOpEnable = VK_TRUE;
    eds3Features.extendedDynamicState3ColorBlendEnable = VK_TRUE;
    eds3Features.extendedDynamicState3ColorBlendEquation = VK_TRUE;
    eds3Features.extendedDynamicState3ColorWriteMask = VK_TRUE;

    vk::PhysicalDeviceVertexInputDynamicStateFeaturesEXT vertexInputFeatures{};
    vertexInputFeatures.vertexInputDynamicState = VK_TRUE;

    // Host-side reset of query pools lets RHITimerQueryVK avoid recording
    // a vkCmdResetQueryPool into each frame's CB. Core in VK 1.2; the
    // EXT struct + extension are still recognized for explicitness.
    vk::PhysicalDeviceHostQueryResetFeatures hostQueryResetFeatures{};
    hostQueryResetFeatures.hostQueryReset = VK_TRUE;

    // Timeline semaphores: backs RHIInteropSync's CUDA↔VK ordering.
    // Core since Vulkan 1.2; the feature must still be opted into.
    vk::PhysicalDeviceTimelineSemaphoreFeatures timelineSemaphoreFeatures{};
    timelineSemaphoreFeatures.timelineSemaphore = VK_TRUE;

    // Chain assembled tail-first so each link's pNext is set before the
    // outer struct points at it. Order in the chain does not matter to
    // the implementation, only that the terminal struct's pNext is null.
    vertexInputFeatures.setPNext(&eds3Features);
    eds3Features.setPNext(&eds2Features);
    eds2Features.setPNext(&eds1Features);
    eds1Features.setPNext(&shaderObjectFeatures);
    shaderObjectFeatures.setPNext(&dynamicRenderingFeatures);
    dynamicRenderingFeatures.setPNext(&hostQueryResetFeatures);
    hostQueryResetFeatures.setPNext(&timelineSemaphoreFeatures);

    vk::DeviceCreateInfo dci{
      vk::DeviceCreateFlags(),
      1, &qci,
      /*enabledLayerCount=*/ 0, /*ppEnabledLayerNames=*/ nullptr,
      uint32_t(kDeviceExtensions.size()), kDeviceExtensions.data(),
      &features};
    dci.setPNext(&vertexInputFeatures);
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

RHIVulkan::ExternalSemaphore RHIVulkan::createExternalSemaphore(vk::SemaphoreType type) const {
  ExternalSemaphore out;

  // Chain order: SemaphoreCreateInfo -> ExportSemaphoreCreateInfo ->
  // SemaphoreTypeCreateInfo (only for timeline; binary uses the default).
  vk::SemaphoreCreateInfo sci{};
  vk::ExportSemaphoreCreateInfo esci{vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd};
  vk::SemaphoreTypeCreateInfo stci{type, /*initialValue=*/ 0};
  sci.setPNext(&esci);
  if (type == vk::SemaphoreType::eTimeline) {
    esci.setPNext(&stci);
  }

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

bool rhiVulkanInstallSingleton(const std::array<uint8_t, VK_UUID_SIZE>& gpuUUID) {
  assert(!s_rhiVk);
  s_rhiVk = RHIVulkan::create(gpuUUID);
  return s_rhiVk != nullptr;
}

void initRHIVulkanInteropContext() {
  std::array<uint8_t, VK_UUID_SIZE> uuid{};
  if (epoxy_has_gl_extension("GL_EXT_memory_object")) {
    glGetUnsignedBytevEXT(GL_DEVICE_UUID_EXT, uuid.data());
  } else {
    fprintf(stderr, "initRHIVulkanInteropContext: GL_EXT_memory_object not available; selecting first VK physical device without UUID match\n");
  }

  if (!rhiVulkanInstallSingleton(uuid)) {
    fprintf(stderr, "initRHIVulkanInteropContext: RHIVulkan::create() failed; CUDA-GL shared surfaces will be unavailable\n");
  }
}
