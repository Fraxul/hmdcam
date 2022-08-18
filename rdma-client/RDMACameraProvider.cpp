#include "RDMACameraProvider.h"
#include "common/VPIUtil.h"
#include "rdma/RDMAContext.h"
#include "rhi/RHI.h"
#include "rhi/cuda/RHICVInterop.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cuda.h>
#include <assert.h>


RDMACameraProvider::RDMACameraProvider(RDMAContext* ctx, SerializationBuffer cfg) : m_rdmaContext(ctx) {

  cfg.rewind();
  m_streamCount = cfg.get_u32();
  m_streamWidth = cfg.get_u32();
  m_streamHeight = cfg.get_u32();

  assert(m_streamCount && m_streamWidth && m_streamHeight);

  size_t rowStride = streamWidth() * 4;
  for (size_t streamIdx = 0; streamIdx < streamCount(); ++streamIdx) {
    char key[32];
    sprintf(key, "camera%zu", streamIdx);
    m_cameraRDMABuffers.push_back(rdmaContext()->newManagedBuffer(std::string(key), rowStride * streamHeight(), kRDMABufferUsageWriteDestination));

    m_cameraSurfaces.push_back(rhi()->newTexture2D(streamWidth(), streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8)));

  }
  m_gpuMatTmp.resize(streamCount());
  m_gpuMatGreyscaleTmp.resize(streamCount());
  m_vpiImages.resize(streamCount());

  // Create initial gpumats before creating VPI wrappers around them
  updateSurfaces();

  for (size_t streamIdx = 0; streamIdx < m_cameraRDMABuffers.size(); ++streamIdx) {
    cv::cuda::GpuMat& gpuMat = m_gpuMatTmp[streamIdx];

    VPIImageData data;
    memset(&data, 0, sizeof(VPIImageData));
    data.bufferType = VPI_IMAGE_BUFFER_CUDA_PITCH_LINEAR;
    data.buffer.pitch.format = VPI_IMAGE_FORMAT_BGRA8;
    data.buffer.pitch.numPlanes = 1;
    data.buffer.pitch.planes[0].pixelType = vpiImageFormatGetPlanePixelType(data.buffer.pitch.format, 0);
    data.buffer.pitch.planes[0].width = m_streamWidth;
    data.buffer.pitch.planes[0].height = m_streamHeight;
    data.buffer.pitch.planes[0].pitchBytes = gpuMat.step;
    data.buffer.pitch.planes[0].data = gpuMat.cudaPtr();

    VPI_CHECK(vpiImageCreateWrapper(&data, /*params=*/ NULL, VPI_BACKEND_CUDA | VPI_REQUIRE_BACKENDS, &m_vpiImages[streamIdx]));
  }

}

RDMACameraProvider::~RDMACameraProvider() {
  for (VPIImage img : m_vpiImages) {
    vpiImageDestroy(img);
  }
}

void RDMACameraProvider::updateSurfaces() {
  if (!m_rdmaBuffersDirty)
    return;

  m_rdmaBuffersDirty = false;

  for (size_t streamIdx = 0; streamIdx < m_cameraRDMABuffers.size(); ++streamIdx) {
    rhi()->loadTextureData(m_cameraSurfaces[streamIdx], kVertexElementTypeUByte4N, m_cameraRDMABuffers[streamIdx]->data());
    m_gpuMatTmp[streamIdx].upload(cvMat(streamIdx));
  }

  m_gpuMatGreyscaleDirty = true;
}

cv::Mat RDMACameraProvider::cvMat(size_t sensorIdx) const {
  return cv::Mat(/*rows=*/ streamHeight(), /*cols=*/ streamWidth(), CV_8UC4, m_cameraRDMABuffers[sensorIdx]->data());
}

cv::cuda::GpuMat RDMACameraProvider::gpuMatGreyscale(size_t sensorIndex) {
  if (m_gpuMatGreyscaleDirty) {
    // demand-populate greyscaled mats
    for (size_t streamIdx = 0; streamIdx < m_cameraRDMABuffers.size(); ++streamIdx) {
      cv::cuda::cvtColor(m_gpuMatTmp[streamIdx], m_gpuMatGreyscaleTmp[streamIdx], cv::COLOR_BGRA2GRAY, 0);
    }
    m_gpuMatGreyscaleDirty = false;
  }

  return m_gpuMatGreyscaleTmp[sensorIndex];
}

VPIImage RDMACameraProvider::vpiImage(size_t sensorIndex) const {
  return m_vpiImages[sensorIndex];
}

/*
void RDMACameraProvider::populateGpuMat(size_t sensorIdx, cv::cuda::GpuMat& gpuMat, const cv::cuda::Stream& stream) {
  gpuMat.upload(cvMat(sensorIdx), const_cast<cv::cuda::Stream&>(stream));
}
*/

