#include "hmdcam/tegra/ArgusCameraSurfaceVK.h"
#include "rhi/RHI.h"
#include "rhi/cuda/CudaUtil.h"
#include <nvbufsurface.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr vk::Format kNV12Format = vk::Format::eG8B8R82Plane420Unorm;

} // namespace

ArgusCameraSurfaceVK::~ArgusCameraSurfaceVK() {
  if (m_cudaLumaTexObject) {
    cuTexObjectDestroy(m_cudaLumaTexObject);
    m_cudaLumaTexObject = 0;
  }
  if (m_cudaChromaTexObject) {
    cuTexObjectDestroy(m_cudaChromaTexObject);
    m_cudaChromaTexObject = 0;
  }
  // Mapped buffers are owned by m_cudaExtMem; destroying the external memory
  // frees both per-plane mappings together.
  if (m_cudaExtMem) {
    cudaDestroyExternalMemory(m_cudaExtMem);
    m_cudaExtMem = nullptr;
  }
  // Base RHISurfaceVK destructor releases m_image / m_memory / m_imageView,
  // which closes the VK-side dup of the FD.

  if (m_surf) {
    NvBufSurfaceDestroy(m_surf);
    m_surf = nullptr;
  }
}

/*static*/ ArgusCameraSurfaceVK::ptr ArgusCameraSurfaceVK::create(NvBufSurface* surf, vk::SamplerYcbcrConversion conversion) {
  if (!surf || surf->numFilled == 0) {
    fprintf(stderr, "ArgusCameraSurfaceVK::create: invalid NvBufSurface\n");
    abort();
  }
  NvBufSurfaceParams& sp = surf->surfaceList[0];
  NvBufSurfacePlaneParams& pp = sp.planeParams;
  if (pp.num_planes != 2) {
    fprintf(stderr, "ArgusCameraSurfaceVK::create: expected NV12 (2 planes), got %u\n", pp.num_planes);
    abort();
  }
  if (sp.colorFormat != NVBUF_COLOR_FORMAT_NV12 && sp.colorFormat != NVBUF_COLOR_FORMAT_NV12_ER) {
    fprintf(stderr, "ArgusCameraSurfaceVK::create: unsupported colorFormat %d (need NV12 or NV12_ER)\n", sp.colorFormat);
    abort();
  }

  RHIVulkan* vk = rhi()->vk();
  if (!vk) {
    fprintf(stderr, "ArgusCameraSurfaceVK::create: rhi()->vk() is null\n");
    abort();
  }
  vk::Device device = vk->device();

  // VkImage extent.width must be padded to match NvBuf's per-row pitch.
  // The driver's per-plane rowPitch under DISJOINT+LINEAR is computed from
  // extent.width * bytesPerPixel; if that's less than NvBuf's hardware-
  // aligned pitch (typical: 1920 logical -> 2048 padded), the driver reads
  // rows at the wrong stride and produces horizontal stripe artifacts.
  //
  // For NV12 the Y plane is 1 byte/pixel, so paddedWidth = pp.pitch[0].
  // (Chroma plane has the same NvBuf pitch, with 2 bytes per UV sample,
  // and NV12 ties plane-1 width to plane-0 width / 2 — the math works out
  // so both planes get matching rowPitch simultaneously.)
  uint32_t extentWidth = pp.pitch[0] / pp.bytesPerPix[0];

  ArgusCameraSurfaceVK::ptr tex(new ArgusCameraSurfaceVK());
  tex->m_surf = surf;
  tex->m_vkFormat = kNV12Format;
  tex->m_rhiFormat = kSurfaceFormat_RGBA8; // sampler delivers RGBA after ycbcr conversion
  // width()/height() report the *valid* (Argus output) dims so consumers'
  // coordinate math is unchanged. extent.width carries the padded dim.
  tex->m_width = sp.width;
  tex->m_height = sp.height;
  tex->m_widthY = pp.width[0];
  tex->m_heightY = pp.height[0];
  tex->m_pitchY = pp.pitch[0];
  tex->m_widthUV = pp.width[1];
  tex->m_heightUV = pp.height[1];
  tex->m_pitchUV = pp.pitch[1];
  tex->m_extentWidth = extentWidth;

  // ------ Create disjoint multi-plane VkImage ------
  // Per the NV12+ycbcr spike: VK_EXT_image_drm_format_modifier is broken on
  // Tegra (no modifiers advertised), so we use LINEAR + DISJOINT + per-plane
  // bind. The driver computes per-plane size/alignment; we bind into NvBuf's
  // layout via memoryOffset on each plane bind.
  vk::ExternalMemoryImageCreateInfo extImgCi;
  extImgCi.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;

  vk::ImageCreateInfo imgCi;
  imgCi.pNext = &extImgCi;
  imgCi.imageType = vk::ImageType::e2D;
  imgCi.format = kNV12Format;
  imgCi.extent = vk::Extent3D{extentWidth, sp.height, 1};
  imgCi.mipLevels = 1;
  imgCi.arrayLayers = 1;
  imgCi.samples = vk::SampleCountFlagBits::e1;
  imgCi.tiling = vk::ImageTiling::eLinear;
  imgCi.usage = vk::ImageUsageFlagBits::eSampled;
  imgCi.sharingMode = vk::SharingMode::eExclusive;
  imgCi.initialLayout = vk::ImageLayout::eUndefined;
  imgCi.flags = vk::ImageCreateFlagBits::eDisjoint;

  tex->m_image = device.createImageUnique(imgCi);

  // Per-plane memory requirements. Sanity-check NvBuf's plane offsets satisfy
  // the driver's alignment requirements; abort on mismatch (would indicate a
  // device or NvBuf version that breaks the spike's assumptions).
  vk::ImageAspectFlagBits planeAspects[2] = {
    vk::ImageAspectFlagBits::ePlane0,
    vk::ImageAspectFlagBits::ePlane1,
  };
  vk::MemoryRequirements planeMemReqs[2];
  uint32_t commonMemTypeBits = 0xffffffffu;
  for (uint32_t i = 0; i < 2; ++i) {
    vk::ImagePlaneMemoryRequirementsInfo planeReqInfo;
    planeReqInfo.planeAspect = planeAspects[i];
    vk::ImageMemoryRequirementsInfo2 req2;
    req2.pNext = &planeReqInfo;
    req2.image = tex->m_image.get();
    vk::MemoryRequirements2 mr2 = device.getImageMemoryRequirements2(req2);
    planeMemReqs[i] = mr2.memoryRequirements;
    commonMemTypeBits &= planeMemReqs[i].memoryTypeBits;
    if (pp.offset[i] % planeMemReqs[i].alignment != 0) {
      fprintf(stderr, "ArgusCameraSurfaceVK: NvBuf plane[%u] offset %u not aligned to VK requirement %llu\n",
        i, pp.offset[i], (unsigned long long) planeMemReqs[i].alignment);
      abort();
    }
  }

  // Intersect with FD's memory-type constraint.
  int vkFd = dup(static_cast<int>(sp.bufferDesc));
  if (vkFd < 0) {
    perror("ArgusCameraSurfaceVK: dup() for VK import");
    abort();
  }
  auto pfn_vkGetMemoryFdPropertiesKHR = reinterpret_cast<PFN_vkGetMemoryFdPropertiesKHR>(
    vk->device().getProcAddr("vkGetMemoryFdPropertiesKHR"));
  if (!pfn_vkGetMemoryFdPropertiesKHR) {
    fprintf(stderr, "ArgusCameraSurfaceVK: vkGetMemoryFdPropertiesKHR not loadable\n");
    abort();
  }
  VkMemoryFdPropertiesKHR fdProps{VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
  if (pfn_vkGetMemoryFdPropertiesKHR(device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, vkFd, &fdProps) != VK_SUCCESS) {
    fprintf(stderr, "ArgusCameraSurfaceVK: vkGetMemoryFdPropertiesKHR failed\n");
    abort();
  }
  uint32_t finalMemTypeBits = commonMemTypeBits & fdProps.memoryTypeBits;
  if (finalMemTypeBits == 0) {
    fprintf(stderr, "ArgusCameraSurfaceVK: no memory type satisfies FD + planes\n");
    abort();
  }
  uint32_t memTypeIndex = vk->findMemoryType(finalMemTypeBits, vk::MemoryPropertyFlags{});

  // Import the FD as one VkDeviceMemory sized to the full NvBuf allocation.
  // No dedicated-allocation flag: both plane binds reference offsets inside
  // this single memory region.
  vk::ImportMemoryFdInfoKHR importInfo;
  importInfo.handleType = vk::ExternalMemoryHandleTypeFlagBits::eDmaBufEXT;
  importInfo.fd = vkFd; // ownership transfers to VK on success
  vk::MemoryAllocateInfo memAi;
  memAi.pNext = &importInfo;
  memAi.allocationSize = sp.dataSize;
  memAi.memoryTypeIndex = memTypeIndex;
  tex->m_memory = device.allocateMemoryUnique(memAi);

  // Bind both planes at the NvBuf-given offsets.
  vk::BindImagePlaneMemoryInfo planeBindInfo[2];
  vk::BindImageMemoryInfo binds[2];
  for (uint32_t i = 0; i < 2; ++i) {
    planeBindInfo[i].planeAspect = planeAspects[i];
    binds[i].pNext = &planeBindInfo[i];
    binds[i].image = tex->m_image.get();
    binds[i].memory = tex->m_memory.get();
    binds[i].memoryOffset = pp.offset[i];
  }
  (void) device.bindImageMemory2(2, binds);

  // Diagnostic: confirm the driver's per-plane row pitch actually matches
  // NvBuf's. With DISJOINT+LINEAR, the driver picks rowPitch from
  // extent.width — typically tight (width * bpp). NvBuf often pads to a
  // hardware-friendly alignment (e.g. 1920 -> 2048). When they differ the
  // sampled image shows horizontal stripes from accumulated row drift.
  static bool s_loggedOnce = false;
  if (!s_loggedOnce) {
    s_loggedOnce = true;
    for (uint32_t i = 0; i < 2; ++i) {
      vk::ImageSubresource sr{planeAspects[i], 0, 0};
      vk::SubresourceLayout lay = device.getImageSubresourceLayout(tex->m_image.get(), sr);
      fprintf(stderr, "ArgusCameraSurfaceVK[diag] plane[%u]: NvBuf pitch=%u VK rowPitch=%llu (%s)\n",
        i, pp.pitch[i], (unsigned long long) lay.rowPitch,
        (lay.rowPitch == pp.pitch[i]) ? "match" : "MISMATCH -- stride artifacts will appear");
    }
  }

  // ImageView with the supplied ycbcr conversion. The aspect for the sampled
  // view is COLOR (covers both planes); the conversion handles the YUV->RGB
  // resolve at sample time.
  vk::SamplerYcbcrConversionInfo viewYcbcrLink;
  viewYcbcrLink.conversion = conversion;
  vk::ImageViewCreateInfo viewCi;
  viewCi.pNext = &viewYcbcrLink;
  viewCi.image = tex->m_image.get();
  viewCi.viewType = vk::ImageViewType::e2D;
  viewCi.format = kNV12Format;
  viewCi.components = vk::ComponentMapping{}; // identity
  viewCi.subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
  tex->m_imageView = device.createImageViewUnique(viewCi);

  // Initial layout transition UNDEFINED -> SHADER_READ_ONLY_OPTIMAL so the
  // first sample works. Argus writes via the EGLImage path (which manages its
  // own layout transitions transparently — the NvBuf doesn't actually have a
  // formal VK layout, the eShaderReadOnlyOptimal request here is the
  // app-side promise to the driver about how the image will be consumed).
  vk::Queue queue = vk->queue();
  uint32_t queueFamily = vk->queueFamilyIndex();
  vk::UniqueCommandPool pool = device.createCommandPoolUnique(
    vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eTransient, queueFamily});
  auto cbs = device.allocateCommandBuffersUnique(
    vk::CommandBufferAllocateInfo{pool.get(), vk::CommandBufferLevel::ePrimary, 1});
  vk::CommandBuffer cb = cbs[0].get();
  cb.begin(vk::CommandBufferBeginInfo{vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  vk::ImageMemoryBarrier toShaderRead{
    vk::AccessFlags(),
    vk::AccessFlagBits::eShaderRead,
    vk::ImageLayout::eUndefined,
    vk::ImageLayout::eShaderReadOnlyOptimal,
    VK_QUEUE_FAMILY_IGNORED,
    VK_QUEUE_FAMILY_IGNORED,
    tex->m_image.get(),
    vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };
  cb.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eFragmentShader,
    vk::DependencyFlags(), 0, nullptr, 0, nullptr, 1, &toShaderRead);
  cb.end();
  vk::SubmitInfo submit;
  submit.setCommandBuffers(cb);
  queue.submit(submit, vk::Fence());
  queue.waitIdle();

  // ------ CUDA side: import the FD and map per-plane buffers ------
  int cudaFd = dup(static_cast<int>(sp.bufferDesc));
  if (cudaFd < 0) {
    perror("ArgusCameraSurfaceVK: dup() for CUDA import");
    abort();
  }
  cudaExternalMemoryHandleDesc memDesc = {};
  memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  memDesc.handle.fd = cudaFd; // ownership transfers to CUDA
  memDesc.size = sp.dataSize;
  CUDA_CHECK(cudaImportExternalMemory(&tex->m_cudaExtMem, &memDesc));

  // Map each plane at its NvBuf offset as a flat buffer of psize bytes.
  // The result is a CUdeviceptr we feed to pitch2D CUtexObjects below.
  cudaExternalMemoryBufferDesc bufY = {};
  bufY.offset = pp.offset[0];
  bufY.size = pp.psize[0];
  void* mappedY = nullptr;
  CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&mappedY, tex->m_cudaExtMem, &bufY));
  tex->m_cudaY = reinterpret_cast<CUdeviceptr>(mappedY);

  cudaExternalMemoryBufferDesc bufUV = {};
  bufUV.offset = pp.offset[1];
  bufUV.size = pp.psize[1];
  void* mappedUV = nullptr;
  CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&mappedUV, tex->m_cudaExtMem, &bufUV));
  tex->m_cudaUV = reinterpret_cast<CUdeviceptr>(mappedUV);

  // Per-plane CUtexObjects matching the existing IArgusCamera contract.
  // Luma plane: 1 channel, format determined by NvBuf bytesPerPix (1 byte for
  // standard NV12).
  CUarray_format lumaFmt = (pp.bytesPerPix[0] == 2) ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT8;
  CUarray_format chromaFmt = (pp.bytesPerPix[1] == 4) ? CU_AD_FORMAT_UNSIGNED_INT16 : CU_AD_FORMAT_UNSIGNED_INT8;

  {
    CUDA_RESOURCE_DESC resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = CU_RESOURCE_TYPE_PITCH2D;
    resDesc.res.pitch2D.devPtr = tex->m_cudaY;
    resDesc.res.pitch2D.format = lumaFmt;
    resDesc.res.pitch2D.numChannels = 1;
    resDesc.res.pitch2D.width = pp.width[0];
    resDesc.res.pitch2D.height = pp.height[0];
    resDesc.res.pitch2D.pitchInBytes = pp.pitch[0];
    CUDA_TEXTURE_DESC texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
    texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    texDesc.maxAnisotropy = 1;
    CUDA_CHECK(cuTexObjectCreate(&tex->m_cudaLumaTexObject, &resDesc, &texDesc, nullptr));
  }
  {
    CUDA_RESOURCE_DESC resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = CU_RESOURCE_TYPE_PITCH2D;
    resDesc.res.pitch2D.devPtr = tex->m_cudaUV;
    resDesc.res.pitch2D.format = chromaFmt;
    resDesc.res.pitch2D.numChannels = 2; // interleaved UV
    resDesc.res.pitch2D.width = pp.width[1];
    resDesc.res.pitch2D.height = pp.height[1];
    resDesc.res.pitch2D.pitchInBytes = pp.pitch[1];
    CUDA_TEXTURE_DESC texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
    texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    texDesc.maxAnisotropy = 1;
    CUDA_CHECK(cuTexObjectCreate(&tex->m_cudaChromaTexObject, &resDesc, &texDesc, nullptr));
  }

  return tex;
}

bool ArgusCameraSurfaceVK::fillCudaMemcpy2D(CUDA_MEMCPY2D& out, bool chromaPlane) const {
  out.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  // Width-in-bytes equals the luma plane width for both planes -- chroma has
  // half the width but two bytes per sample (interleaved UV), so the byte
  // count of a row is the same. Matches the pre-migration behaviour.
  out.WidthInBytes = m_widthY;
  if (chromaPlane) {
    out.srcDevice = m_cudaUV;
    out.srcPitch = m_pitchUV;
    out.Height = m_heightUV;
  } else {
    out.srcDevice = m_cudaY;
    out.srcPitch = m_pitchY;
    out.Height = m_heightY;
  }
  return true;
}
