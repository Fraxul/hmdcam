#pragma once
#include "common/ICameraProvider.h"
#include "common/SerializationBuffer.h"
#include "rhi/RHISurface.h"
#include <vector>
#include <opencv2/core/mat.hpp>
#ifdef HAVE_VPI2
#include <vpi/Image.h>
#endif
#include "rhi/gl/GLCommon.h" // must be included before cudaEGL
#include <cudaEGL.h>

class DebugCameraProvider : public ICameraProvider {
public:
  DebugCameraProvider();
  virtual ~DebugCameraProvider();

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

  bool connect(const char* hostname);
  void updateSurfaces();

protected:
  int m_fd = -1;

  static void* streamThreadEntryPoint(void* x) { reinterpret_cast<DebugCameraProvider*>(x)->streamThreadFn(); return NULL; }
  void streamThreadFn();
  pthread_t m_streamThread = 0;
  pthread_mutex_t m_frameConsumedMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t m_frameConsumedCond = PTHREAD_COND_INITIALIZER;

  // Start with this flag set so that updateSurfaces will create allocations during init
  bool m_streamFrameReadyToConsume = true;

  size_t m_streamCount = 0;
  unsigned int m_streamWidth = 0, m_streamHeight = 0;

  CUeglColorFormat m_eglColorFormat;
  CUDA_RESOURCE_DESC m_lumaResourceDescriptor;
  CUDA_RESOURCE_DESC m_chromaResourceDescriptor;
  uint32_t m_lumaPlaneSizeBytes = 0, m_chromaPlaneSizeBytes = 0;

  struct StreamData {
    void* hostLumaBuffer = nullptr;
    void* hostChromaBuffer = nullptr;

    RHISurface::ptr rhiSurfaceRGBA;
    cv::cuda::GpuMat gpuMatLuma, gpuMatChroma, gpuMatRGBA;
    CUtexObject cudaLumaTexObject = 0;
    VPIImage vpiImage = nullptr;
  };

  std::vector<StreamData> m_streamData;

  cv::cuda::GpuMat m_gpuMatRGBTmp; // format conversion intermediary
};

