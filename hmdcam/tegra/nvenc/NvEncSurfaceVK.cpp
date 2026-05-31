#include "hmdcam/tegra/nvenc/NvEncSurfaceVK.h"
#include "rhi/RHI.h"
#include <nvbufsurface.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
constexpr vk::Format kRGBAFormat = vk::Format::eR8G8B8A8Unorm;
} // namespace

NvEncSurfaceVK::~NvEncSurfaceVK() {
  // Base RHISurfaceVK destructor releases m_image / m_memory / m_imageView,
  // which closes the VK-side dup of the DMA-BUF FD. Then free the NvBuf.
  if (m_surf) {
    NvBufSurfaceDestroy(m_surf);
    m_surf = nullptr;
  }
}

/*static*/ NvEncSurfaceVK::ptr NvEncSurfaceVK::create(uint32_t width, uint32_t height) {
  // ------ Allocate the RGBA pitch-linear NvBufSurface (VIC input) ------
  NvBufSurfaceAllocateParams allocParams;
  memset(&allocParams, 0, sizeof(allocParams));
  allocParams.params.width = width;
  allocParams.params.height = height;
  allocParams.params.layout = NVBUF_LAYOUT_PITCH;
  allocParams.params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  allocParams.params.memType = NVBUF_MEM_SURFACE_ARRAY;
  allocParams.memtag = NvBufSurfaceTag_VIDEO_CONVERT;

  NvBufSurface* surf = nullptr;
  if (NvBufSurfaceAllocate(&surf, 1, &allocParams) != 0 || !surf) {
    fprintf(stderr, "NvEncSurfaceVK::create: NvBufSurfaceAllocate failed for %ux%u RGBA\n", width, height);
    abort();
  }
  surf->numFilled = 1;

  NvBufSurfaceParams& sp = surf->surfaceList[0];
  NvBufSurfacePlaneParams& pp = sp.planeParams;
  if (pp.num_planes != 1) {
    fprintf(stderr, "NvEncSurfaceVK::create: expected single-plane RGBA, got %u planes\n", pp.num_planes);
    abort();
  }

  RHIVulkan* vk = rhi()->vk();
  if (!vk) {
    fprintf(stderr, "NvEncSurfaceVK::create: rhi()->vk() is null (VK backend required)\n");
    abort();
  }
  vk::Device device = vk->device();

  // VkImage extent.width is padded to NvBuf's per-row pitch so the driver's
  // rowPitch matches NvBuf's hardware-aligned pitch (RGBA -> 4 bytes/pixel).
  uint32_t extentWidth = pp.pitch[0] / pp.bytesPerPix[0];

  NvEncSurfaceVK::ptr tex(new NvEncSurfaceVK());
  tex->m_surf = surf;
  tex->m_vkFormat = kRGBAFormat;
  tex->m_rhiFormat = kSurfaceFormat_RGBA8;
  // width()/height() report the logical (requested) dims; extent.width carries
  // the padded dim. RHIRenderTargetVK uses width()/height() for the render
  // area, so we draw into the logical [0, width()) sub-region.
  tex->m_width = sp.width;
  tex->m_height = sp.height;
  tex->m_extentWidth = extentWidth;

  // ------ Create the renderable VkImage over the imported DMA-BUF ------
  vk::ExternalMemoryImageCreateInfo extImgCi;
  extImgCi.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;

  vk::ImageCreateInfo imgCi;
  imgCi.pNext = &extImgCi;
  imgCi.imageType = vk::ImageType::e2D;
  imgCi.format = kRGBAFormat;
  imgCi.extent = vk::Extent3D{extentWidth, sp.height, 1};
  imgCi.mipLevels = 1;
  imgCi.arrayLayers = 1;
  imgCi.samples = vk::SampleCountFlagBits::e1;
  imgCi.tiling = vk::ImageTiling::eLinear;
  // COLOR_ATTACHMENT is the whole point (direct render). TRANSFER_SRC is added
  // so the RHI debug-dump path could snapshot it if ever needed; both are
  // cheap and don't constrain the linear import.
  imgCi.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
  imgCi.sharingMode = vk::SharingMode::eExclusive;
  imgCi.initialLayout = vk::ImageLayout::eUndefined;
  tex->m_image = device.createImageUnique(imgCi);

  // Memory requirements for the (single-plane) image.
  vk::MemoryRequirements memReq = device.getImageMemoryRequirements(tex->m_image.get());
  if (pp.offset[0] % memReq.alignment != 0) {
    fprintf(stderr, "NvEncSurfaceVK::create: NvBuf plane offset %u not aligned to VK requirement %llu\n",
      pp.offset[0], (unsigned long long) memReq.alignment);
    abort();
  }

  // Intersect the image's memory-type constraint with the FD's.
  int vkFd = dup(static_cast<int>(sp.bufferDesc));
  if (vkFd < 0) {
    perror("NvEncSurfaceVK::create: dup() for VK import");
    abort();
  }
  auto pfn_vkGetMemoryFdPropertiesKHR = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
    device.getProcAddr("vkGetMemoryFdPropertiesKHR"));
  if (!pfn_vkGetMemoryFdPropertiesKHR) {
    fprintf(stderr, "NvEncSurfaceVK::create: vkGetMemoryFdPropertiesKHR not loadable\n");
    abort();
  }
  VkMemoryFdPropertiesKHR fdProps{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
  if (pfn_vkGetMemoryFdPropertiesKHR(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, vkFd, &fdProps) != VK_SUCCESS) {
    fprintf(stderr, "NvEncSurfaceVK::create: vkGetMemoryFdPropertiesKHR failed\n");
    abort();
  }
  uint32_t finalMemTypeBits = memReq.memoryTypeBits & fdProps.memoryTypeBits;
  if (finalMemTypeBits == 0) {
    fprintf(stderr, "NvEncSurfaceVK::create: no memory type satisfies FD + image\n");
    abort();
  }
  uint32_t memTypeIndex = vk->findMemoryType(finalMemTypeBits, vk::MemoryPropertyFlags{});

  // Import the FD as one VkDeviceMemory sized to the full NvBuf allocation.
  vk::ImportMemoryFdInfoKHR importInfo;
  importInfo.handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;
  importInfo.fd = vkFd; // ownership transfers to VK on success
  vk::MemoryAllocateInfo memAi;
  memAi.pNext = &importInfo;
  memAi.allocationSize = sp.dataSize;
  memAi.memoryTypeIndex = memTypeIndex;
  tex->m_memory = device.allocateMemoryUnique(memAi);

  device.bindImageMemory(tex->m_image.get(), tex->m_memory.get(), pp.offset[0]);

  // Diagnostic: confirm the driver's rowPitch matches NvBuf's. A mismatch
  // produces stride artifacts (diagonal/striped image). Logged once.
  {
    vk::ImageSubresource sr{vk::ImageAspectFlagBits::eColor, 0, 0};
    vk::SubresourceLayout lay = device.getImageSubresourceLayout(tex->m_image.get(), sr);
    fprintf(stderr, "NvEncSurfaceVK::create: %ux%u (extent.width=%u) NvBuf pitch=%u VK rowPitch=%llu (%s)\n",
      sp.width, sp.height, extentWidth, pp.pitch[0], (unsigned long long) lay.rowPitch,
      (lay.rowPitch == pp.pitch[0]) ? "match" : "MISMATCH -- stride artifacts will appear");
  }

  // Color image view (no ycbcr).
  vk::ImageViewCreateInfo viewCi;
  viewCi.image = tex->m_image.get();
  viewCi.viewType = vk::ImageViewType::e2D;
  viewCi.format = kRGBAFormat;
  viewCi.components = vk::ComponentMapping{}; // identity
  viewCi.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
  tex->m_imageView = device.createImageViewUnique(viewCi);

  // No initial layout transition: beginRenderPass transitions UNDEFINED ->
  // COLOR_ATTACHMENT_OPTIMAL on each pass (the debug pass uses kLoadInvalidate,
  // which discards prior contents anyway).

  return tex;
}
