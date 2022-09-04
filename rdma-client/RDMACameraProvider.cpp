#include "RDMACameraProvider.h"
#include "common/VPIUtil.h"
#include "rdma/RDMAContext.h"
#include "rhi/RHI.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/cuda/RHICVInterop.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda.h>
#include <assert.h>
#include <npp.h>


RDMACameraProvider::RDMACameraProvider(RDMAContext* ctx, SerializationBuffer cfg) : m_rdmaContext(ctx) {

  cfg.rewind();
  m_streamCount = cfg.get_u32();
  m_streamWidth = cfg.get_u32();
  m_streamHeight = cfg.get_u32();
  m_eglColorFormat = (CUeglColorFormat) cfg.get_u32();

  memset(&m_lumaResourceDescriptor, 0, sizeof(m_lumaResourceDescriptor));
  m_lumaResourceDescriptor.resType = CU_RESOURCE_TYPE_PITCH2D;
  m_lumaResourceDescriptor.res.pitch2D.format = (CUarray_format) cfg.get_u32();
  m_lumaResourceDescriptor.res.pitch2D.numChannels = cfg.get_u32();
  m_lumaResourceDescriptor.res.pitch2D.width = cfg.get_u32();
  m_lumaResourceDescriptor.res.pitch2D.height = cfg.get_u32();
  m_lumaResourceDescriptor.res.pitch2D.pitchInBytes = cfg.get_u32();

  memset(&m_chromaResourceDescriptor, 0, sizeof(m_chromaResourceDescriptor));
  m_chromaResourceDescriptor.resType = CU_RESOURCE_TYPE_PITCH2D;
  m_chromaResourceDescriptor.res.pitch2D.format = (CUarray_format) cfg.get_u32();
  m_chromaResourceDescriptor.res.pitch2D.numChannels = cfg.get_u32();
  m_chromaResourceDescriptor.res.pitch2D.width = cfg.get_u32();
  m_chromaResourceDescriptor.res.pitch2D.height = cfg.get_u32();
  m_chromaResourceDescriptor.res.pitch2D.pitchInBytes = cfg.get_u32();

  printf("  Luma: %.4zu x %.4zu NumChannels=%u Format=%x pitchInBytes=%zu\n",
    m_lumaResourceDescriptor.res.pitch2D.width, m_lumaResourceDescriptor.res.pitch2D.height,
    m_lumaResourceDescriptor.res.pitch2D.numChannels,
    m_lumaResourceDescriptor.res.pitch2D.format,
    m_lumaResourceDescriptor.res.pitch2D.pitchInBytes);
  printf("Chroma: %.4zu x %.4zu NumChannels=%u Format=%x pitchInBytes=%zu\n",
    m_chromaResourceDescriptor.res.pitch2D.width, m_chromaResourceDescriptor.res.pitch2D.height,
    m_chromaResourceDescriptor.res.pitch2D.numChannels,
    m_chromaResourceDescriptor.res.pitch2D.format,
    m_chromaResourceDescriptor.res.pitch2D.pitchInBytes);

  assert(m_streamCount && m_streamWidth && m_streamHeight);

  // Allocate format conversion temporary
  m_gpuMatRGBTmp.create(m_streamHeight, m_streamWidth, CV_8UC3);

  // Setup RDMA buffers and RHISurfaces
  m_streamData.resize(streamCount());
  for (size_t streamIdx = 0; streamIdx < streamCount(); ++streamIdx) {
    StreamData& sd = m_streamData[streamIdx];

    char key[32];
    sprintf(key, "camera%zu_luma", streamIdx);
    sd.rdmaLumaBuffer = rdmaContext()->newManagedBuffer(std::string(key),
      m_lumaResourceDescriptor.res.pitch2D.pitchInBytes * m_lumaResourceDescriptor.res.pitch2D.height, kRDMABufferUsageWriteDestination);

    sprintf(key, "camera%zu_chroma", streamIdx);
    sd.rdmaChromaBuffer = rdmaContext()->newManagedBuffer(std::string(key),
      m_chromaResourceDescriptor.res.pitch2D.pitchInBytes * m_chromaResourceDescriptor.res.pitch2D.height, kRDMABufferUsageWriteDestination);

    sd.rhiSurfaceRGBA = rhi()->newTexture2D(streamWidth(), streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
  }

  // Ensure gpumats are created
  updateSurfaces();

  // Wrap GpuMats to create CUtexObjects
  for (size_t streamIdx = 0; streamIdx < streamCount(); ++streamIdx) {
    StreamData& sd = m_streamData[streamIdx];
    // Luma array and texobject

    {
      m_lumaResourceDescriptor.res.pitch2D.devPtr = (CUdeviceptr) sd.gpuMatLuma.cudaPtr();

      CUDA_TEXTURE_DESC texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
      texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
      texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
      texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
      // texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES; // optional
      texDesc.maxAnisotropy = 1;

      CUDA_CHECK(cuTexObjectCreate(&sd.cudaLumaTexObject, &m_lumaResourceDescriptor, &texDesc, /*resourceViewDescriptor=*/ nullptr));
    }
  }

#ifdef HAVE_VPI2
  // Wrap GpuMats to create VPIImages
  for (size_t streamIdx = 0; streamIdx < m_streamCount; ++streamIdx) {
    StreamData& sd = m_streamData[streamIdx];

    VPIImageData data;
    memset(&data, 0, sizeof(VPIImageData));
    data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    data.buffer.pitch.format = VPI_IMAGE_FORMAT_Y8_ER;
    data.buffer.pitch.numPlanes = 1;
    data.buffer.pitch.planes[0].pixelType = vpiImageFormatGetPlanePixelType(data.buffer.pitch.format, 0);
    data.buffer.pitch.planes[0].width = m_streamWidth;
    data.buffer.pitch.planes[0].height = m_streamHeight;
    data.buffer.pitch.planes[0].pitchBytes = sd.gpuMatLuma.step;
    data.buffer.pitch.planes[0].data = sd.gpuMatLuma.cudaPtr();

    VPI_CHECK(vpiImageCreateWrapper(&data, /*params=*/ NULL, VPI_BACKEND_CUDA | VPI_REQUIRE_BACKENDS, &sd.vpiImage));
  }
#endif
}

RDMACameraProvider::~RDMACameraProvider() {
  for (StreamData& sd : m_streamData) {
    cuTexObjectDestroy(sd.cudaLumaTexObject);
#ifdef HAVE_VPI2
    vpiImageDestroy(sd.vpiImage);
#endif
  }
}

void RDMACameraProvider::updateSurfaces() {
  if (!m_rdmaBuffersDirty)
    return;

  m_rdmaBuffersDirty = false;

  for (size_t streamIdx = 0; streamIdx < m_streamCount; ++streamIdx) {
    StreamData& sd = m_streamData[streamIdx];

    sd.gpuMatLuma.upload(cvMatLuma(streamIdx));
    sd.gpuMatChroma.upload(cvMatChroma(streamIdx));

    // Ensure RGB[A] mats are allocated before color conversion
    sd.gpuMatRGBA.create(m_streamHeight, m_streamWidth, CV_8UC4);

    Npp8u const* pSrc[] = {(const Npp8u*) sd.gpuMatLuma.cudaPtr(), (const Npp8u*) sd.gpuMatChroma.cudaPtr()};

    NppiSize sz;
    sz.width = m_streamWidth;
    sz.height = m_streamHeight;
    assert(sd.gpuMatLuma.step == sd.gpuMatChroma.step);

    // No direct NV12 to RGBA conversion function, so we chain NV12 to RGB and then RGB to RGBA
    nppiNV12ToRGB_709HDTV_8u_P2C3R(pSrc, sd.gpuMatLuma.step, (Npp8u*) m_gpuMatRGBTmp.cudaPtr(), m_gpuMatRGBTmp.step, sz);
    int destinationOrder[] = {0, 1, 2, 3};
    nppiSwapChannels_8u_C3C4R((const Npp8u*) m_gpuMatRGBTmp.cudaPtr(), m_gpuMatRGBTmp.step,
      (Npp8u*) sd.gpuMatRGBA.cudaPtr(), sd.gpuMatRGBA.step,
      sz, destinationOrder, /*constant (source channel 3) value=*/ 0xff);

    RHICUDA::copyGpuMatToSurface(sd.gpuMatRGBA, sd.rhiSurfaceRGBA);
  }

}

cv::Mat RDMACameraProvider::cvMatLuma(size_t streamIdx) const {
  // TODO support more formats
  return cv::Mat(/*rows=*/ m_lumaResourceDescriptor.res.pitch2D.height, /*cols=*/ m_lumaResourceDescriptor.res.pitch2D.width, CV_8U,
    m_streamData[streamIdx].rdmaLumaBuffer->data(), m_lumaResourceDescriptor.res.pitch2D.pitchInBytes);
}

cv::Mat RDMACameraProvider::cvMatChroma(size_t streamIdx) const {
  // TODO support more formats
  return cv::Mat(/*rows=*/ m_chromaResourceDescriptor.res.pitch2D.height, /*cols=*/ m_chromaResourceDescriptor.res.pitch2D.width, CV_8UC2,
    m_streamData[streamIdx].rdmaChromaBuffer->data(), m_chromaResourceDescriptor.res.pitch2D.pitchInBytes);
}

cv::cuda::GpuMat RDMACameraProvider::gpuMatGreyscale(size_t streamIdx) {
  return m_streamData[streamIdx].gpuMatLuma;
}

VPIImage RDMACameraProvider::vpiImage(size_t streamIdx) const {
#if HAVE_VPI2
  return m_streamData[streamIdx].vpiImage;
#else
  assert(false && "vpiImageGreyscale: HAVE_VPI2 was not defined at compile time");
  return nullptr;
#endif
}

RHISurface::ptr RDMACameraProvider::rgbTexture(size_t streamIdx) const {
  return m_streamData[streamIdx].rhiSurfaceRGBA;
}

CUtexObject RDMACameraProvider::cudaLumaTexObject(size_t streamIdx) const {
  return m_streamData[streamIdx].cudaLumaTexObject;
}

