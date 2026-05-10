#pragma once
// RHIInteropSyncGL: A synchronization mechanism between GL and CUDA.
#include "rhi/RHIObject.h"
#include "rhi/gl/GLCommon.h"
#include "rhi/vk/RHIVulkan.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <set>
#include <vector>

class RHIInteropSync : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIInteropSync> ptr;
  RHIInteropSync();
  virtual ~RHIInteropSync();

  // Sync barrier functions.

  // Call this when you're done with CUDA work and will be switching to RHI
  void signalCUDAToRHI(CUstream stream);

  // Call this when you're done with RHI work and will be switching to CUDA
  void signalRHItoCUDA(CUstream stream);

  void addSyncedObject(RHIObject*);
  void removeSyncedObject(RHIObject*);

protected:
  std::set<RHIObject*> m_syncedObjects;

  void rebuildSyncIDLists();

  std::vector<GLuint> m_waitBufferIDs;
  std::vector<GLuint> m_signalBufferIDs;

  std::vector<GLuint> m_waitTextureIDs;
  std::vector<GLenum> m_waitTextureSrcLayouts;

  std::vector<GLuint> m_signalTextureIDs;
  std::vector<GLenum> m_signalTextureDstLayouts;

  RHIVulkan::ExternalSemaphore m_cudaToGlSem;
  RHIVulkan::ExternalSemaphore m_glToCudaSem;
  GLuint m_cudaToGlSemGL = 0;
  GLuint m_glToCudaSemGL = 0;
  cudaExternalSemaphore_t m_cudaToGlSemCU = nullptr;
  cudaExternalSemaphore_t m_glToCudaSemCU = nullptr;
};
