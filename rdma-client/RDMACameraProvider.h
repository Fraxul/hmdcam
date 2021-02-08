#pragma once
#include "common/ICameraProvider.h"
#include "rdma/RDMABuffer.h"
#include "rdma/SerializationBuffer.h"
#include "rhi/RHISurface.h"
#include <vector>
#include <opencv2/core/mat.hpp>

class RDMAContext;

class RDMACameraProvider : public ICameraProvider {
public:
  RDMACameraProvider(RDMAContext*, SerializationBuffer config);
  virtual ~RDMACameraProvider();

  virtual size_t streamCount() const { return m_streamCount; }
  virtual unsigned int streamWidth() const { return m_streamWidth; }
  virtual unsigned int streamHeight() const { return m_streamHeight; }
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const { return m_cameraSurfaces[sensorIndex]; }
  cv::Mat cvMat(size_t sensorIndex) const;

  void flagRDMABuffersDirty() { m_rdmaBuffersDirty = true; }
  void updateSurfaces();

protected:
  RDMAContext* m_rdmaContext;
  RDMAContext* rdmaContext() const { return m_rdmaContext; }

  size_t m_streamCount;
  unsigned int m_streamWidth, m_streamHeight;

  std::vector<RDMABuffer::ptr> m_cameraRDMABuffers;
  std::vector<RHISurface::ptr> m_cameraSurfaces;


  bool m_rdmaBuffersDirty;

};

