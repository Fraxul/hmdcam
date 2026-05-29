#include "rhi/vk/RHIInteropSurfaceVK.h"
#include "rhi/RHI.h"
#include "rhi/cuda/CudaUtil.h"
#include <stdio.h>
#include <unistd.h>

namespace {

struct FormatMap {
  vk::Format vkFormat;
  cudaChannelFormatDesc cudaFormat;
};

FormatMap rhiFormatToInteropMap(RHISurfaceFormat f) {
  // Channel format kind only matters for surf2Dread/Write interpretation;
  // raw memcpys go through bits unchanged. Matches the table in
  // RHIInteropSurfaceGL.cpp — kept in sync intentionally so both interop
  // surface paths produce identical CUDA mappings.
  using FK = cudaChannelFormatKind;
  const FK U = cudaChannelFormatKindUnsigned;
  const FK S = cudaChannelFormatKindSigned;
  const FK F = cudaChannelFormatKindFloat;
  switch (f) {
    case kSurfaceFormat_R8: return {vk::Format::eR8Unorm, cudaCreateChannelDesc(8, 0, 0, 0, U)};
    case kSurfaceFormat_R8i: return {vk::Format::eR8Sint, cudaCreateChannelDesc(8, 0, 0, 0, S)};
    case kSurfaceFormat_R8ui: return {vk::Format::eR8Uint, cudaCreateChannelDesc(8, 0, 0, 0, U)};
    case kSurfaceFormat_R16: return {vk::Format::eR16Unorm, cudaCreateChannelDesc(16, 0, 0, 0, U)};
    case kSurfaceFormat_R16f: return {vk::Format::eR16Sfloat, cudaCreateChannelDesc(16, 0, 0, 0, F)};
    case kSurfaceFormat_R16i: return {vk::Format::eR16Sint, cudaCreateChannelDesc(16, 0, 0, 0, S)};
    case kSurfaceFormat_R16ui: return {vk::Format::eR16Uint, cudaCreateChannelDesc(16, 0, 0, 0, U)};
    case kSurfaceFormat_RG8i: return {vk::Format::eR8G8Sint, cudaCreateChannelDesc(8, 8, 0, 0, S)};
    case kSurfaceFormat_RG8ui: return {vk::Format::eR8G8Uint, cudaCreateChannelDesc(8, 8, 0, 0, U)};
    case kSurfaceFormat_RGBA8: return {vk::Format::eR8G8B8A8Unorm, cudaCreateChannelDesc(8, 8, 8, 8, U)};
    case kSurfaceFormat_sRGB8_A8: return {vk::Format::eR8G8B8A8Srgb, cudaCreateChannelDesc(8, 8, 8, 8, U)};
    case kSurfaceFormat_RG16: return {vk::Format::eR16G16Unorm, cudaCreateChannelDesc(16, 16, 0, 0, U)};
    case kSurfaceFormat_R32f: return {vk::Format::eR32Sfloat, cudaCreateChannelDesc(32, 0, 0, 0, F)};
    case kSurfaceFormat_R32i: return {vk::Format::eR32Sint, cudaCreateChannelDesc(32, 0, 0, 0, S)};
    case kSurfaceFormat_R32ui: return {vk::Format::eR32Uint, cudaCreateChannelDesc(32, 0, 0, 0, U)};
    case kSurfaceFormat_RG16i: return {vk::Format::eR16G16Sint, cudaCreateChannelDesc(16, 16, 0, 0, S)};
    case kSurfaceFormat_RG16ui: return {vk::Format::eR16G16Uint, cudaCreateChannelDesc(16, 16, 0, 0, U)};
    case kSurfaceFormat_RGBA8i: return {vk::Format::eR8G8B8A8Sint, cudaCreateChannelDesc(8, 8, 8, 8, S)};
    case kSurfaceFormat_RGBA8ui: return {vk::Format::eR8G8B8A8Uint, cudaCreateChannelDesc(8, 8, 8, 8, U)};
    case kSurfaceFormat_RGBA16f: return {vk::Format::eR16G16B16A16Sfloat, cudaCreateChannelDesc(16, 16, 16, 16, F)};
    case kSurfaceFormat_RGBA16i: return {vk::Format::eR16G16B16A16Sint, cudaCreateChannelDesc(16, 16, 16, 16, S)};
    case kSurfaceFormat_RGBA16ui: return {vk::Format::eR16G16B16A16Uint, cudaCreateChannelDesc(16, 16, 16, 16, U)};
    case kSurfaceFormat_RG32f: return {vk::Format::eR32G32Sfloat, cudaCreateChannelDesc(32, 32, 0, 0, F)};
    case kSurfaceFormat_RG32i: return {vk::Format::eR32G32Sint, cudaCreateChannelDesc(32, 32, 0, 0, S)};
    case kSurfaceFormat_RG32ui: return {vk::Format::eR32G32Uint, cudaCreateChannelDesc(32, 32, 0, 0, U)};
    case kSurfaceFormat_RGBA32i: return {vk::Format::eR32G32B32A32Sint, cudaCreateChannelDesc(32, 32, 32, 32, S)};
    case kSurfaceFormat_RGBA32ui: return {vk::Format::eR32G32B32A32Uint, cudaCreateChannelDesc(32, 32, 32, 32, U)};
    default:
      fprintf(stderr, "RHIInteropSurfaceVK: unsupported RHI surface format %d for interop\n", static_cast<int>(f));
      abort();
  }
}

} // namespace

RHIInteropSurfaceVK::RHIInteropSurfaceVK() :
  RHISurfaceVK() {}

RHIInteropSurfaceVK::~RHIInteropSurfaceVK() {
  // m_cudaArray is a level of m_cudaMipmappedArray and is implicitly
  // released when the mipmapped array is destroyed.
  m_cudaArray = nullptr;
  if (m_cudaMipmappedArray) {
    cudaFreeMipmappedArray(m_cudaMipmappedArray);
    m_cudaMipmappedArray = nullptr;
  }
  if (m_cudaExtMem) {
    cudaDestroyExternalMemory(m_cudaExtMem);
    m_cudaExtMem = nullptr;
  }
  // Base RHISurfaceVK destructor releases m_image / m_memory / m_imageView.
}

/*static*/ RHIInteropSurfaceVK* RHIInteropSurfaceVK::newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor& descriptor) {
  RHIVulkan* vk = rhi()->vk();
  if (!vk) {
    fprintf(stderr, "RHIInteropSurfaceVK::newTexture2D: rhi()->vk() is null; cannot allocate interop surface\n");
    abort();
  }
  if (descriptor.createMips || descriptor.createArray || descriptor.layers > 1 || descriptor.samples > 1) {
    fprintf(stderr, "RHIInteropSurfaceVK::newTexture2D: descriptor flags (mips/array/layers/samples) not supported for interop surfaces\n");
    abort();
  }

  FormatMap fmt = rhiFormatToInteropMap(descriptor.format);

  RHIInteropSurfaceVK* tex = new RHIInteropSurfaceVK();
  tex->m_rhiFormat = descriptor.format;
  tex->m_vkFormat = fmt.vkFormat;
  tex->m_width = width;
  tex->m_height = height;

  // VK side: exportable optimal-tiled image. Permissive usage so the VK
  // backend can bind the surface anywhere it likes.
  vk::ImageUsageFlags usage =
    vk::ImageUsageFlagBits::eSampled |
    vk::ImageUsageFlagBits::eStorage |
    vk::ImageUsageFlagBits::eColorAttachment |
    vk::ImageUsageFlagBits::eTransferSrc |
    vk::ImageUsageFlagBits::eTransferDst;
  RHIVulkan::ExternalImage external = vk->allocateExternalImage(width, height, fmt.vkFormat, usage, vk::ImageTiling::eOptimal);

  // Move the externally-allocated image + memory into the base RHISurfaceVK
  // fields so vkImage() / vkImageView() (used by RHIVK loadTexture) return
  // the same handle CUDA imports below.
  tex->m_image = std::move(external.image);
  tex->m_memory = std::move(external.memory);

  vk::Device device = vk->device();
  vk::ImageViewCreateInfo viewCi{
    vk::ImageViewCreateFlags(),
    tex->m_image.get(),
    vk::ImageViewType::e2D,
    fmt.vkFormat,
    vk::ComponentMapping{ },
    vk::ImageSubresourceRange{ vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };
  tex->m_imageView = device.createImageViewUnique(viewCi);

  // The image starts in eUndefined layout. CUDA writes via cudaArray bypass
  // VK's layout machinery, but the first VK *sample* needs the image in
  // SHADER_READ_ONLY (or GENERAL). Push a one-shot UNDEFINED → GENERAL
  // transition so subsequent VK reads see a defined layout. Using GENERAL
  // matches the GL-interop path which uses GL_LAYOUT_GENERAL_EXT for
  // CUDA-shared surfaces.
  vk::Queue queue = vk->queue();
  uint32_t queueFamily = vk->queueFamilyIndex();
  vk::CommandPoolCreateInfo poolCi{vk::CommandPoolCreateFlagBits::eTransient, queueFamily};
  vk::UniqueCommandPool pool = device.createCommandPoolUnique(poolCi);
  vk::CommandBufferAllocateInfo cbAi{pool.get(), vk::CommandBufferLevel::ePrimary, 1};
  auto cbs = device.allocateCommandBuffersUnique(cbAi);
  vk::CommandBuffer cb = cbs[0].get();
  cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  vk::ImageMemoryBarrier toGeneral{
    vk::AccessFlags(), vk::AccessFlagBits::eShaderRead,
    vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
    tex->m_image.get(),
    vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
  };
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eFragmentShader,
    vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &toGeneral);
  cb.end();
  vk::SubmitInfo submit;
  submit.setCommandBuffers(cb);
  queue.submit(submit, vk::Fence());
  queue.waitIdle();

  // CUDA side: import memory and map as a mipmapped array with surface
  // load/store enabled. Mirrors RHIInteropSurfaceGL exactly.
  cudaExternalMemoryHandleDesc memDesc = {};
  memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  memDesc.handle.fd = external.memoryFd; // CUDA takes ownership
  memDesc.size = external.allocationSize;
  if (external.isDedicated)
    memDesc.flags |= cudaExternalMemoryDedicated;
  CUDA_CHECK(cudaImportExternalMemory(&tex->m_cudaExtMem, &memDesc));
  external.memoryFd = -1;

  cudaExternalMemoryMipmappedArrayDesc arrDesc = {};
  arrDesc.offset = 0;
  arrDesc.formatDesc = fmt.cudaFormat;
  arrDesc.extent = make_cudaExtent(width, height, 0); // 2D non-layered
  arrDesc.flags = cudaArraySurfaceLoadStore;
  arrDesc.numLevels = 1;
  CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&tex->m_cudaMipmappedArray, tex->m_cudaExtMem, &arrDesc));
  CUDA_CHECK(cudaGetMipmappedArrayLevel(&tex->m_cudaArray, tex->m_cudaMipmappedArray, 0));

  return tex;
}

bool RHIInteropSurfaceVK::isInteropSurface() const {
  return true;
}

cudaArray_t RHIInteropSurfaceVK::cudaArray() const {
  return m_cudaArray;
}
