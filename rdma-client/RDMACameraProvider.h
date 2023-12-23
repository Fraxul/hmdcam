#pragma once
#include "common/ICameraProvider.h"
#include "rdma/RDMABuffer.h"
#include "rdma/SerializationBuffer.h"
#include "rhi/RHISurface.h"
#include <vector>
#include <opencv2/core/mat.hpp>
#ifdef HAVE_VPI2
#include <vpi/Image.h>
#endif
#include "rhi/gl/GLCommon.h" // must be included before cudaEGL
#include <cudaEGL.h>

class RDMAContext;

class RDMACameraProvider : public ICameraProvider {
public:
  RDMACameraProvider(RDMAContext*, SerializationBuffer config);
  virtual ~RDMACameraProvider();

  virtual size_t streamCount() const { return m_streamCount; }
  virtual unsigned int streamWidth() const { return m_streamWidth; }
  virtual unsigned int streamHeight() const { return m_streamHeight; }
  virtual RHISurface::ptr rgbTexture(size_t streamIdx) const;
  virtual const char* rgbTextureGLSamplerType() const { return "sampler2D"; }
  virtual CUtexObject cudaLumaTexObject(size_t streamIdx) const;
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t streamIdx);
  virtual VPIImage vpiImage(size_t streamIdx) const;
  cv::Mat cvMatLuma(size_t streamIdx) const;
  cv::Mat cvMatChroma(size_t streamIdx) const;

  void flagRDMABuffersDirty() { m_rdmaBuffersDirty = true; }
  void updateSurfaces();

protected:
  RDMAContext* m_rdmaContext;
  RDMAContext* rdmaContext() const { return m_rdmaContext; }

  size_t m_streamCount;
  unsigned int m_streamWidth, m_streamHeight;

  CUeglColorFormat m_eglColorFormat;
  CUDA_RESOURCE_DESC m_lumaResourceDescriptor;
  CUDA_RESOURCE_DESC m_chromaResourceDescriptor;
  uint32_t m_lumaCopyWidthBytes, m_chromaCopyWidthBytes;

  struct StreamData {
    RDMABuffer::ptr rdmaLumaBuffer;
    RDMABuffer::ptr rdmaChromaBuffer;

    RHISurface::ptr rhiSurfaceRGBA;
    cv::cuda::GpuMat gpuMatLuma, gpuMatChroma, gpuMatRGBA;
    CUtexObject cudaLumaTexObject = 0;
    VPIImage vpiImage = nullptr;
  };

  std::vector<StreamData> m_streamData;

  cv::cuda::GpuMat m_gpuMatRGBTmp; // format conversion intermediary

  bool m_rdmaBuffersDirty = true;
};

