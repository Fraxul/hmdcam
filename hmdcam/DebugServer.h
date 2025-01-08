#pragma once
#include <pthread.h>
#include <stdint.h>
#include <vector>
#include <cuda.h>
#include <boost/noncopyable.hpp>
#include "common/SerializationBuffer.h"
#include <opencv2/core.hpp>
#include <opencv2/core/cvstd.hpp>

class CameraSystem;
class IArgusCamera;
class DepthMapGenerator;

class DebugServer {
public:

  DebugServer();
  ~DebugServer();
  bool initWithCameraSystem(CameraSystem*, IArgusCamera*, DepthMapGenerator*);

  // Called in main-thead frame processing loop -- copies buffers for Tx if necessary
  void frameProcessingEnded();

  bool hasConnection() { return m_streamConnected; }


protected:
  CameraSystem* m_cameraSystem;
  IArgusCamera* m_cameraProvider;
  DepthMapGenerator* m_depthMapGenerator;


  static void* streamThreadEntryPoint(void* x) { reinterpret_cast<DebugServer*>(x)->streamThreadFn(); return NULL; }
  void streamThreadFn();
  pthread_t m_streamThread = 0;
  pthread_mutex_t m_streamReadyMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t m_streamReadyCond = PTHREAD_COND_INITIALIZER;

  bool m_streamConnected = false;

  bool m_streamReadyForNextFrame = false;

  CUDA_RESOURCE_DESC m_lumaResourceDescriptor;
  CUDA_RESOURCE_DESC m_chromaResourceDescriptor;

  struct StreamResource : boost::noncopyable {
    StreamResource() {}

    ~StreamResource() {
      if (m_lumaPlane)
        cuMemFreeHost(m_lumaPlane);
      if (m_chromaPlane)
        cuMemFreeHost(m_chromaPlane);
    }

    void* m_lumaPlane = nullptr;
    void* m_chromaPlane = nullptr;
  };

  StreamResource* m_streamResources = nullptr;
  uint32_t m_streamCount = 0;
  uint32_t m_lumaPlaneSizeBytes = 0;
  uint32_t m_chromaPlaneSizeBytes = 0;

  // L/R inputs to disparity algo
  std::vector<cv::Mat> m_disparityInputStreams[2];
  uint32_t m_disparityInputStreamSizeBytes = 0;

  std::vector<cv::Mat> m_disparityStreams;
  uint32_t m_disparityStreamSizeBytes = 0;

  SerializationBuffer m_streamHeader;
  cv::String m_cameraSystemConfig;

};

