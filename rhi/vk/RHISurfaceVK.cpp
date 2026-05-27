#include "rhi/vk/RHISurfaceVK.h"
#include "rhi/RHI.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

vk::Format rhiSurfaceFormatToVK(RHISurfaceFormat f) {
  switch (f) {
    case kSurfaceFormat_R8: return vk::Format::eR8Unorm;
    case kSurfaceFormat_R8i: return vk::Format::eR8Sint;
    case kSurfaceFormat_R8ui: return vk::Format::eR8Uint;
    case kSurfaceFormat_R16: return vk::Format::eR16Unorm;
    case kSurfaceFormat_R16f: return vk::Format::eR16Sfloat;
    case kSurfaceFormat_R16i: return vk::Format::eR16Sint;
    case kSurfaceFormat_R16ui: return vk::Format::eR16Uint;
    case kSurfaceFormat_R32f: return vk::Format::eR32Sfloat;
    case kSurfaceFormat_R32i: return vk::Format::eR32Sint;
    case kSurfaceFormat_R32ui: return vk::Format::eR32Uint;
    case kSurfaceFormat_RG8i: return vk::Format::eR8G8Sint;
    case kSurfaceFormat_RG8ui: return vk::Format::eR8G8Uint;
    case kSurfaceFormat_RG16: return vk::Format::eR16G16Unorm;
    case kSurfaceFormat_RG16i: return vk::Format::eR16G16Sint;
    case kSurfaceFormat_RG16ui: return vk::Format::eR16G16Uint;
    case kSurfaceFormat_RG32f: return vk::Format::eR32G32Sfloat;
    case kSurfaceFormat_RG32i: return vk::Format::eR32G32Sint;
    case kSurfaceFormat_RG32ui: return vk::Format::eR32G32Uint;
    case kSurfaceFormat_RGBA8: return vk::Format::eR8G8B8A8Unorm;
    case kSurfaceFormat_sRGB8_A8: return vk::Format::eR8G8B8A8Srgb;
    case kSurfaceFormat_RGB10_A2: return vk::Format::eA2B10G10R10UnormPack32;
    case kSurfaceFormat_RGBA8i: return vk::Format::eR8G8B8A8Sint;
    case kSurfaceFormat_RGBA8ui: return vk::Format::eR8G8B8A8Uint;
    case kSurfaceFormat_RGBA16f: return vk::Format::eR16G16B16A16Sfloat;
    case kSurfaceFormat_RGBA16i: return vk::Format::eR16G16B16A16Sint;
    case kSurfaceFormat_RGBA16ui: return vk::Format::eR16G16B16A16Uint;
    case kSurfaceFormat_RGBA16s: return vk::Format::eR16G16B16A16Snorm;
    case kSurfaceFormat_RGBA32i: return vk::Format::eR32G32B32A32Sint;
    case kSurfaceFormat_RGBA32ui: return vk::Format::eR32G32B32A32Uint;
    case kSurfaceFormat_Depth16: return vk::Format::eD16Unorm;
    case kSurfaceFormat_Depth32f: return vk::Format::eD32Sfloat;
    case kSurfaceFormat_Depth32f_Stencil8: return vk::Format::eD32SfloatS8Uint;
    case kSurfaceFormat_Stencil8: return vk::Format::eS8Uint;
    default:
      fprintf(stderr, "rhiSurfaceFormatToVK: unsupported format %d\n", static_cast<int>(f));
      abort();
  }
}

RHISurfaceVK::~RHISurfaceVK() {}

/*static*/ RHISurfaceVK::ptr RHISurfaceVK::newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor& descriptor) {
  if (descriptor.createMips || descriptor.createArray || descriptor.layers > 1 || descriptor.samples > 1) {
    // TODO: We currently support the basic 2D case only, pending application use of the more complex texture types.
    fprintf(stderr, "RHISurfaceVK::newTexture2D: descriptor (createMips=%d createArray=%d layers=%u samples=%u) not yet supported\n",
      descriptor.createMips, descriptor.createArray, descriptor.layers, descriptor.samples);
    abort();
  }

  RHISurfaceVK::ptr s(new RHISurfaceVK());
  s->m_rhiFormat = descriptor.format;
  s->m_vkFormat = rhiSurfaceFormatToVK(descriptor.format);
  s->m_width = width;
  s->m_height = height;

  vk::Device device = rhi()->vk()->device();

  // Permissive usage: we don't know up front whether the surface will be
  // sampled, drawn to, or used as a transfer src/dst. Including everything
  // costs nothing on the driver side beyond a bit of validation.
  vk::ImageUsageFlags usage =
    vk::ImageUsageFlagBits::eSampled |
    vk::ImageUsageFlagBits::eStorage |
    vk::ImageUsageFlagBits::eColorAttachment |
    vk::ImageUsageFlagBits::eTransferSrc |
    vk::ImageUsageFlagBits::eTransferDst;
  if (rhiSurfaceFormatHasDepth(descriptor.format) || rhiSurfaceFormatHasStencil(descriptor.format)) {
    usage = (usage & ~(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage)) | vk::ImageUsageFlagBits::eDepthStencilAttachment;
  }

  vk::ImageCreateInfo ici{
    vk::ImageCreateFlags(),
    vk::ImageType::e2D,
    s->m_vkFormat,
    vk::Extent3D(width, height, 1),
    /*mipLevels=*/ 1,
    /*arrayLayers=*/ 1,
    vk::SampleCountFlagBits::e1,
    vk::ImageTiling::eOptimal,
    usage,
    vk::SharingMode::eExclusive,
  };
  s->m_image = device.createImageUnique(ici);

  vk::MemoryRequirements memReq = device.getImageMemoryRequirements(s->m_image.get());
  uint32_t memTypeIndex = rhi()->vk()->findMemoryType(
    memReq.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
  vk::MemoryAllocateInfo allocInfo{memReq.size, memTypeIndex};
  s->m_memory = device.allocateMemoryUnique(allocInfo);
  device.bindImageMemory(s->m_image.get(), s->m_memory.get(), 0);

  vk::ImageAspectFlags aspect = (rhiSurfaceFormatHasDepth(descriptor.format) || rhiSurfaceFormatHasStencil(descriptor.format))
    ? (rhiSurfaceFormatHasStencil(descriptor.format)
          ? (vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil)
          : vk::ImageAspectFlagBits::eDepth)
    : vk::ImageAspectFlagBits::eColor;
  vk::ImageViewCreateInfo viewCi{
    vk::ImageViewCreateFlags(),
    s->m_image.get(),
    vk::ImageViewType::e2D,
    s->m_vkFormat,
    vk::ComponentMapping{ },
    vk::ImageSubresourceRange{ aspect, 0, 1, 0, 1},
  };
  s->m_imageView = device.createImageViewUnique(viewCi);

  return s;
}

void RHISurfaceVK::loadFromCPU(const void* data, size_t bytes) {
  vk::Device device = rhi()->vk()->device();
  uint32_t queueFamily = rhi()->vk()->queueFamilyIndex();
  vk::Queue queue = rhi()->vk()->queue();

  // Staging buffer (host-visible).
  vk::BufferCreateInfo bufCi{vk::BufferCreateFlags(), bytes, vk::BufferUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive};
  vk::UniqueBuffer stagingBuf = device.createBufferUnique(bufCi);
  vk::MemoryRequirements stagingReq = device.getBufferMemoryRequirements(stagingBuf.get());
  uint32_t stagingMemType = rhi()->vk()->findMemoryType(
    stagingReq.memoryTypeBits,
    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
  vk::UniqueDeviceMemory stagingMem = device.allocateMemoryUnique({stagingReq.size, stagingMemType});
  device.bindBufferMemory(stagingBuf.get(), stagingMem.get(), 0);

  void* mapped = device.mapMemory(stagingMem.get(), 0, bytes);
  memcpy(mapped, data, bytes);
  device.unmapMemory(stagingMem.get());

  // One-shot command buffer for the upload + layout transitions.
  vk::CommandPoolCreateInfo poolCi{vk::CommandPoolCreateFlagBits::eTransient, queueFamily};
  vk::UniqueCommandPool pool = device.createCommandPoolUnique(poolCi);
  vk::CommandBufferAllocateInfo cbAi{pool.get(), vk::CommandBufferLevel::ePrimary, 1};
  auto cbs = device.allocateCommandBuffersUnique(cbAi);
  vk::CommandBuffer cb = cbs[0].get();

  cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  vk::ImageSubresourceRange range{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

  // UNDEFINED -> TRANSFER_DST_OPTIMAL
  vk::ImageMemoryBarrier toTransfer{
    vk::AccessFlags(), vk::AccessFlagBits::eTransferWrite,
    vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
    m_image.get(), range};
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer,
    vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &toTransfer);

  vk::BufferImageCopy copy{
    0, 0, 0,
    vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor,        0, 0, 1},
    vk::Offset3D{                              0,        0, 0  },
    vk::Extent3D{                        m_width, m_height, 1  }
  };
  cb.copyBufferToImage(stagingBuf.get(), m_image.get(), vk::ImageLayout::eTransferDstOptimal, 1, &copy);

  // TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL (most common sampled use)
  vk::ImageMemoryBarrier toSampled{
    vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
    vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
    m_image.get(), range};
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader,
    vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &toSampled);

  cb.end();

  // TODO: Synchronous submit -- this function is not expected to be in the hot path right now.
  // We can add batched upload later if necessary.
  vk::SubmitInfo submit;
  submit.setCommandBuffers(cb);
  queue.submit(submit, vk::Fence());
  queue.waitIdle();
}

CUgraphicsResource& RHISurfaceVK::cuGraphicsResource() const {
  fprintf(stderr, "RHISurfaceVK::cuGraphicsResource: not implemented (no CUDA interop yet)\n");
  abort();
}
