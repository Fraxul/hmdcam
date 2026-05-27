#include "rhi/vk/RHIInteropSurfaceGL.h"
#include "rhi/RHI.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/gl/GLCommon.h"
#include <epoxy/egl.h>
#include <stdio.h>
#include <unistd.h>

namespace {
// Phase 4 transitional: when running on the VK RHI backend, no GL context
// is current. Skip the GL-side memory import in that case — CUDA + VK
// halves of the surface are still constructed; only m_glId remains 0 (and
// any GL consumer would fail loudly if it tried to use it).
bool hasCurrentGLContext() { return eglGetCurrentContext() != EGL_NO_CONTEXT; }
} // namespace

namespace {

struct FormatMap {
  vk::Format vkFormat;
  cudaChannelFormatDesc cudaFormat;
  uint32_t bytesPerPixel;
};

FormatMap rhiFormatToInteropMap(RHISurfaceFormat f) {
  // Channel format kind controls how surf2Dread / surf2Dwrite interpret the
  // bits. For raw memcpy paths (cudaMemcpy2DToArrayAsync) this is mostly
  // cosmetic — the bits go through unchanged.
  using FK = cudaChannelFormatKind;
  const FK U = cudaChannelFormatKindUnsigned;
  const FK S = cudaChannelFormatKindSigned;
  const FK F = cudaChannelFormatKindFloat;
  switch (f) {
    case kSurfaceFormat_R8: return {vk::Format::eR8Unorm, cudaCreateChannelDesc(8, 0, 0, 0, U), 1};
    case kSurfaceFormat_R8i: return {vk::Format::eR8Sint, cudaCreateChannelDesc(8, 0, 0, 0, S), 1};
    case kSurfaceFormat_R8ui: return {vk::Format::eR8Uint, cudaCreateChannelDesc(8, 0, 0, 0, U), 1};

    case kSurfaceFormat_R16: return {vk::Format::eR16Unorm, cudaCreateChannelDesc(16, 0, 0, 0, U), 2};
    case kSurfaceFormat_R16f: return {vk::Format::eR16Sfloat, cudaCreateChannelDesc(16, 0, 0, 0, F), 2};
    case kSurfaceFormat_R16i: return {vk::Format::eR16Sint, cudaCreateChannelDesc(16, 0, 0, 0, S), 2};
    case kSurfaceFormat_R16ui: return {vk::Format::eR16Uint, cudaCreateChannelDesc(16, 0, 0, 0, U), 2};
    case kSurfaceFormat_RG8i: return {vk::Format::eR8G8Sint, cudaCreateChannelDesc(8, 8, 0, 0, S), 2};
    case kSurfaceFormat_RG8ui: return {vk::Format::eR8G8Uint, cudaCreateChannelDesc(8, 8, 0, 0, U), 2};

    case kSurfaceFormat_RGBA8: return {vk::Format::eR8G8B8A8Unorm, cudaCreateChannelDesc(8, 8, 8, 8, U), 4};
    case kSurfaceFormat_sRGB8_A8: return {vk::Format::eR8G8B8A8Srgb, cudaCreateChannelDesc(8, 8, 8, 8, U), 4};
    case kSurfaceFormat_RG16: return {vk::Format::eR16G16Unorm, cudaCreateChannelDesc(16, 16, 0, 0, U), 4};
    case kSurfaceFormat_R32f: return {vk::Format::eR32Sfloat, cudaCreateChannelDesc(32, 0, 0, 0, F), 4};
    case kSurfaceFormat_R32i: return {vk::Format::eR32Sint, cudaCreateChannelDesc(32, 0, 0, 0, S), 4};
    case kSurfaceFormat_R32ui: return {vk::Format::eR32Uint, cudaCreateChannelDesc(32, 0, 0, 0, U), 4};
    case kSurfaceFormat_RG16i: return {vk::Format::eR16G16Sint, cudaCreateChannelDesc(16, 16, 0, 0, S), 4};
    case kSurfaceFormat_RG16ui: return {vk::Format::eR16G16Uint, cudaCreateChannelDesc(16, 16, 0, 0, U), 4};
    case kSurfaceFormat_RGBA8i: return {vk::Format::eR8G8B8A8Sint, cudaCreateChannelDesc(8, 8, 8, 8, S), 4};
    case kSurfaceFormat_RGBA8ui: return {vk::Format::eR8G8B8A8Uint, cudaCreateChannelDesc(8, 8, 8, 8, U), 4};

    case kSurfaceFormat_RGBA16f: return {vk::Format::eR16G16B16A16Sfloat, cudaCreateChannelDesc(16, 16, 16, 16, F), 8};
    case kSurfaceFormat_RGBA16i: return {vk::Format::eR16G16B16A16Sint, cudaCreateChannelDesc(16, 16, 16, 16, S), 8};
    case kSurfaceFormat_RGBA16ui: return {vk::Format::eR16G16B16A16Uint, cudaCreateChannelDesc(16, 16, 16, 16, U), 8};
    case kSurfaceFormat_RG32f: return {vk::Format::eR32G32Sfloat, cudaCreateChannelDesc(32, 32, 0, 0, F), 8};
    case kSurfaceFormat_RG32i: return {vk::Format::eR32G32Sint, cudaCreateChannelDesc(32, 32, 0, 0, S), 8};
    case kSurfaceFormat_RG32ui: return {vk::Format::eR32G32Uint, cudaCreateChannelDesc(32, 32, 0, 0, U), 8};

    case kSurfaceFormat_RGBA32i: return {vk::Format::eR32G32B32A32Sint, cudaCreateChannelDesc(32, 32, 32, 32, S), 16};
    case kSurfaceFormat_RGBA32ui: return {vk::Format::eR32G32B32A32Uint, cudaCreateChannelDesc(32, 32, 32, 32, U), 16};

    default:
      fprintf(stderr, "rhiFormatToInteropMap: unsupported RHI surface format %d for interop\n", static_cast<int>(f));
      abort();
  }
}

} // namespace

RHIInteropSurfaceGL::RHIInteropSurfaceGL() :
  RHISurfaceGL() {}

RHIInteropSurfaceGL::~RHIInteropSurfaceGL() {
  // CUDA-side teardown
  if (m_cudaSurfaceObject) {
    cudaDestroySurfaceObject(m_cudaSurfaceObject);
    m_cudaSurfaceObject = 0;
  }
  if (m_cudaTextureObject) {
    cudaDestroyTextureObject(m_cudaTextureObject);
    m_cudaTextureObject = 0;
  }
  // m_cudaArray is a level of m_cudaMipmappedArray and doesn't need to be
  // freed independently — destroying the mipmapped array releases everything.
  m_cudaArray = nullptr;
  if (m_cudaMipmappedArray) {
    cudaFreeMipmappedArray(m_cudaMipmappedArray);
    m_cudaMipmappedArray = nullptr;
  }
  if (m_cudaExtMem) {
    cudaDestroyExternalMemory(m_cudaExtMem);
    m_cudaExtMem = nullptr;
  }

  // GL-side teardown
  if (m_glMemoryObject && hasCurrentGLContext()) {
    glDeleteMemoryObjectsEXT(1, &m_glMemoryObject);
    m_glMemoryObject = 0;
  }
  // Base RHISurfaceGL destructor deletes m_glId (the texture handle).
}

/*static*/ RHIInteropSurfaceGL* RHIInteropSurfaceGL::newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor& descriptor, const RHIInteropSyncDescriptor& syncDescriptor) {
  RHIVulkan* vk = rhi()->vk();
  if (!vk) {
    fprintf(stderr, "RHIInteropSurfaceGL::newTexture2D: rhi()->vk() is null; cannot allocate interop surface\n");
    abort();
  }
  // The interop surface is a single 2D image with no mips, no layers, no MS.
  if (descriptor.createMips || descriptor.createArray || descriptor.layers > 1 || descriptor.samples > 1) {
    fprintf(stderr, "RHIInteropSurfaceGL::newTexture2D: descriptor flags (mips/array/layers/samples) not supported for interop surfaces\n");
    abort();
  }

  FormatMap fmt = rhiFormatToInteropMap(descriptor.format);

  RHIInteropSurfaceGL* tex = new RHIInteropSurfaceGL();
  tex->m_syncDescriptor = syncDescriptor;
  tex->m_rhiFormat = descriptor.format;
  tex->m_glInternalFormat = RHISurfaceFormatToGL(descriptor.format);
  tex->m_glTarget = GL_TEXTURE_2D;
  tex->m_width = width;
  tex->m_height = height;
  tex->m_depth = 1;
  tex->m_layers = 1;
  tex->m_samples = 1;
  tex->m_levels = 1;
  tex->m_isArrayTexture = false;

  // VK side: optimal-tiled image, exportable backing memory. Optimal tiling
  // is required because Tegra's GL_EXT_memory_object does not accept
  // GL_LINEAR_TILING_EXT for imported memory.
  vk::ImageUsageFlags usage =
    vk::ImageUsageFlagBits::eSampled |
    vk::ImageUsageFlagBits::eStorage |
    vk::ImageUsageFlagBits::eColorAttachment |
    vk::ImageUsageFlagBits::eTransferSrc |
    vk::ImageUsageFlagBits::eTransferDst;
  tex->m_vkImage = vk->allocateExternalImage(width, height, fmt.vkFormat, usage, vk::ImageTiling::eOptimal);

  // GL side: import memory and create the texture on top of it. Tiling is
  // left at the GL default (optimal) to match the VK side. Skipped when no
  // GL context is current (VK RHI backend) — m_glId stays 0 and GL
  // consumers (if any) would error loudly on use.
  if (hasCurrentGLContext()) {
    GL(glGenTextures(1, &tex->m_glId));
    GL(glBindTexture(GL_TEXTURE_2D, tex->m_glId));

    GL(glCreateMemoryObjectsEXT(1, &tex->m_glMemoryObject));
    if (tex->m_vkImage.isDedicated) {
      GLint dedicated = GL_TRUE;
      GL(glMemoryObjectParameterivEXT(tex->m_glMemoryObject, GL_DEDICATED_MEMORY_OBJECT_EXT, &dedicated));
    }

    int memFdForGL = dup(tex->m_vkImage.memoryFd);
    GL(glImportMemoryFdEXT(tex->m_glMemoryObject, tex->m_vkImage.allocationSize, GL_HANDLE_TYPE_OPAQUE_FD_EXT, memFdForGL));
    GL(glTexStorageMem2DEXT(GL_TEXTURE_2D, /*levels=*/ 1, tex->m_glInternalFormat, width, height, tex->m_glMemoryObject, /*offset=*/ 0));

    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL(glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  }

  // CUDA side: import memory and map as a mipmapped array. cudaArraySurfaceLoadStore
  // is required so we can wrap level 0 in a cudaSurfaceObject.
  cudaExternalMemoryHandleDesc memDesc = {};
  memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  memDesc.handle.fd = tex->m_vkImage.memoryFd; // CUDA takes ownership
  memDesc.size = tex->m_vkImage.allocationSize;
  if (tex->m_vkImage.isDedicated)
    memDesc.flags |= cudaExternalMemoryDedicated;
  CUDA_CHECK(cudaImportExternalMemory(&tex->m_cudaExtMem, &memDesc));
  tex->m_vkImage.memoryFd = -1;

  cudaExternalMemoryMipmappedArrayDesc arrDesc = {};
  arrDesc.offset = 0;
  arrDesc.formatDesc = fmt.cudaFormat;
  arrDesc.extent = make_cudaExtent(width, height, 0); // 2D non-layered
  arrDesc.flags = cudaArraySurfaceLoadStore;
  arrDesc.numLevels = 1;
  CUDA_CHECK(cudaExternalMemoryGetMappedMipmappedArray(&tex->m_cudaMipmappedArray, tex->m_cudaExtMem, &arrDesc));
  CUDA_CHECK(cudaGetMipmappedArrayLevel(&tex->m_cudaArray, tex->m_cudaMipmappedArray, 0));

  // Eagerly create surface and texture objects bound to the array.
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = tex->m_cudaArray;
  CUDA_CHECK(cudaCreateSurfaceObject(&tex->m_cudaSurfaceObject, &resDesc));

  cudaTextureDesc texDesc = {};
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModePoint;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  CUDA_CHECK(cudaCreateTextureObject(&tex->m_cudaTextureObject, &resDesc, &texDesc, /*resViewDesc=*/ nullptr));

  return tex;
}

void RHIInteropSurfaceGL::copyFromGpuMatAsync(const cv::cuda::GpuMat& src, cudaStream_t stream) {
  size_t copyWidth = std::min<size_t>(m_width, static_cast<size_t>(src.cols));
  size_t copyHeight = std::min<size_t>(m_height, static_cast<size_t>(src.rows));

  CUDA_CHECK(cudaMemcpy2DToArrayAsync(
    m_cudaArray, /*wOffsetBytes=*/ 0, /*hOffset=*/ 0,
    src.cudaPtr(), src.step,
    copyWidth * src.elemSize(), copyHeight,
    cudaMemcpyDeviceToDevice, stream));
}
