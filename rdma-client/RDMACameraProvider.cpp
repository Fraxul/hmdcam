#include "RDMACameraProvider.h"
#include "rdma/RDMAContext.h"
#include "rhi/RHI.h"
#include <opencv2/core/mat.hpp>


RDMACameraProvider::RDMACameraProvider(RDMAContext* ctx, SerializationBuffer cfg) : m_rdmaContext(ctx), m_rdmaBuffersDirty(true) {

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
}

RDMACameraProvider::~RDMACameraProvider() {

}

void RDMACameraProvider::updateSurfaces() {
  if (!m_rdmaBuffersDirty)
    return;

  m_rdmaBuffersDirty = false;

  for (size_t streamIdx = 0; streamIdx < m_cameraRDMABuffers.size(); ++streamIdx) {
    rhi()->loadTextureData(m_cameraSurfaces[streamIdx], kVertexElementTypeUByte4N, m_cameraRDMABuffers[streamIdx]->data());
  }
}

cv::Mat RDMACameraProvider::cvMat(size_t sensorIdx) const {
  return cv::Mat(m_streamHeight, m_streamWidth, CV_8UC4, m_cameraRDMABuffers[sensorIdx]->data());
}

