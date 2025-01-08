#pragma once
#include "common/DepthMapGenerator.h"
#include "common/ICameraProvider.h"
#include "common/SerializationBuffer.h"
#include "rhi/RHISurface.h"
#include <vector>
#include <opencv2/core/mat.hpp>
#include "rhi/gl/GLCommon.h" // must be included before cudaEGL
#include <cudaEGL.h>

class DebugCameraProvider : public ICameraProvider, public DepthMapGenerator {
public:
  DebugCameraProvider();
  virtual ~DebugCameraProvider();

  // -- ICameraProvider
  virtual size_t streamCount() const { return m_streamCount; }
  virtual unsigned int streamWidth() const { return m_streamWidth; }
  virtual unsigned int streamHeight() const { return m_streamHeight; }
  virtual RHISurface::ptr rgbTexture(size_t streamIdx) const;
  virtual const char* rgbTextureGLSamplerType() const { return "sampler2D"; }
  virtual CUtexObject cudaLumaTexObject(size_t streamIdx) const;
  virtual CUtexObject cudaChromaTexObject(size_t streamIdx) const;
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t streamIdx);

  // -- DepthMapGenerator

  bool depthMapGeneratorEnabled() const { return (m_stereoViewCount != 0); }
  virtual void internalLoadSettings(cv::FileStorage&);
  virtual void internalSaveSettings(cv::FileStorage&);
  virtual void internalProcessFrame();
  virtual void internalRenderIMGUI();
  virtual void internalRenderIMGUIPerformanceGraphs();


  // -- Other DebugCameraProvider-specific functions

  void setCudaGLInteropOK(bool value) { m_doCudaGLInterop = value; }
  cv::Mat cvMatLuma(size_t streamIdx) const;
  cv::Mat cvMatChroma(size_t streamIdx) const;
  bool connect(const char* hostname);
  void updateSurfaces();

  const cv::String& cameraSystemConfig() const { return m_cameraSystemConfig; }

protected:
  bool m_doCudaGLInterop = true;
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
  cv::String m_cameraSystemConfig;
  cv::cuda::GpuMat m_gpuMatRGBTmp; // format conversion intermediary

  // CameraProvider stream data

  struct StreamData {
    void* hostLumaBuffer = nullptr;
    void* hostChromaBuffer = nullptr;

    RHISurface::ptr rhiSurfaceRGBA;
    cv::cuda::GpuMat gpuMatLuma, gpuMatChroma, gpuMatRGBA;

    void* hostRGBABuffer = nullptr; // used on fallback path
    CUtexObject cudaLumaTexObject = 0;
    CUtexObject cudaChromaTexObject = 0;
  };

  std::vector<StreamData> m_streamData;

  // -- DepthMapGenerator view data

  struct ViewDataDebug : public ViewData {
    ViewDataDebug() {}
    virtual ~ViewDataDebug() {}

    cv::Mat receivedDisparityInput[2];
    cv::Mat receivedDisparity;
  };

  virtual ViewData* newEmptyViewData() { return new ViewDataDebug(); }
  virtual void internalUpdateViewData();
  ViewDataDebug* viewDataAtIndex(size_t index) { return static_cast<ViewDataDebug*>(m_viewData[index]); }

  uint32_t m_stereoViewCount = 0;
  uint32_t m_stereoDisparityInputSizeBytes = 0;
  uint32_t m_stereoDisparitySizeBytes = 0;
  uint32_t m_disparityWidth = 0, m_disparityHeight = 0;

  std::vector<cv::Mat> m_stereoDisparityInputRecvMats[2];
  std::vector<cv::Mat> m_stereoDisparityRecvMats;
};

