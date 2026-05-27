#include "rhi/vk/RHIInteropSync.h"
#include "rhi/vk/RHIInteropBufferGL.h"
#include "rhi/vk/RHIInteropBufferVK.h"
#include "rhi/vk/RHIInteropSurfaceGL.h"
#include "rhi/RHI.h"
#include "rhi/cuda/CudaUtil.h"
#include <epoxy/egl.h>

RHIInteropSync::RHIInteropSync() {
  RHIVulkan* vk = rhi()->vk();
  if (!vk) {
    fprintf(stderr, "RHIInteropSync: rhi()->vk() is null; cannot allocate interop sync\n");
    abort();
  }

  // Detect whether a GL/EGL context is current. The VK RHI backend runs
  // without GL, in which case the GL semaphore-import path doesn't apply.
  // Phase 4 transitional: skip GL-side setup; CUDA-side signals become
  // no-op pairs (no GL consumer to wait on them). Phase 6 will replace
  // this whole class with timeline-semaphore CUDA↔VK sync.
  m_hasGLContext = (eglGetCurrentContext() != EGL_NO_CONTEXT);

  // Sync primitives: one binary semaphore per direction.
  m_cudaToGlSem = vk->createExternalSemaphore();
  m_glToCudaSem = vk->createExternalSemaphore();

  if (m_hasGLContext) {
    // GL side: import semaphores.
    GL(glGenSemaphoresEXT(1, &m_cudaToGlSemGL));
    int cudaToGlFdForGL = dup(m_cudaToGlSem.fd);
    GL(glImportSemaphoreFdEXT(m_cudaToGlSemGL, GL_HANDLE_TYPE_OPAQUE_FD_EXT, cudaToGlFdForGL));

    GL(glGenSemaphoresEXT(1, &m_glToCudaSemGL));
    int glToCudaFdForGL = dup(m_glToCudaSem.fd);
    GL(glImportSemaphoreFdEXT(m_glToCudaSemGL, GL_HANDLE_TYPE_OPAQUE_FD_EXT, glToCudaFdForGL));
  } else {
    printf("RHIInteropSync: no GL context current; running in CUDA-only stub mode (Phase 4 transitional)\n");
  }

  // CUDA side: import semaphores.
  cudaExternalSemaphoreHandleDesc cudaToGlSemDesc = {};
  cudaToGlSemDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  cudaToGlSemDesc.handle.fd = m_cudaToGlSem.fd;
  CUDA_CHECK(cudaImportExternalSemaphore(&m_cudaToGlSemCU, &cudaToGlSemDesc));
  m_cudaToGlSem.fd = -1;

  cudaExternalSemaphoreHandleDesc glToCudaSemDesc = {};
  glToCudaSemDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
  glToCudaSemDesc.handle.fd = m_glToCudaSem.fd;
  CUDA_CHECK(cudaImportExternalSemaphore(&m_glToCudaSemCU, &glToCudaSemDesc));
  m_glToCudaSem.fd = -1;
}

RHIInteropSync::~RHIInteropSync() {
  // CUDA-side teardown
  if (m_cudaToGlSemCU) {
    cudaDestroyExternalSemaphore(m_cudaToGlSemCU);
    m_cudaToGlSemCU = nullptr;
  }
  if (m_glToCudaSemCU) {
    cudaDestroyExternalSemaphore(m_glToCudaSemCU);
    m_glToCudaSemCU = nullptr;
  }

  // GL-side teardown
  if (m_cudaToGlSemGL) {
    glDeleteSemaphoresEXT(1, &m_cudaToGlSemGL);
    m_cudaToGlSemGL = 0;
  }
  if (m_glToCudaSemGL) {
    glDeleteSemaphoresEXT(1, &m_glToCudaSemGL);
    m_glToCudaSemGL = 0;
  }
}

void RHIInteropSync::addSyncedObject(RHIObject* object) {
  m_syncedObjects.insert(object);
  rebuildSyncIDLists();
}

void RHIInteropSync::removeSyncedObject(RHIObject* object) {
  auto it = m_syncedObjects.find(object);
  assert(it != m_syncedObjects.end());
  m_syncedObjects.erase(it);
  rebuildSyncIDLists();
}

void RHIInteropSync::rebuildSyncIDLists() {
  m_waitBufferIDs.clear();
  m_signalBufferIDs.clear();
  m_waitTextureIDs.clear();
  m_waitTextureSrcLayouts.clear();
  m_signalTextureIDs.clear();
  m_signalTextureDstLayouts.clear();

  for (RHIObject* obj : m_syncedObjects) {
    {
      RHIInteropBufferGL* buf = dynamic_cast<RHIInteropBufferGL*>(obj);
      if (buf) {
        GLuint bufferID = buf->glId();
        RHIInteropSyncDescriptor desc = buf->syncDescriptor();

        if (desc.direction() & kSyncDirectionCUDAWriter) {
          m_waitBufferIDs.push_back(bufferID);
        }
        if (desc.direction() & kSyncDirectionRHIWriter) {
          m_signalBufferIDs.push_back(bufferID);
        }

        continue;
      }
    }
    {
      RHIInteropSurfaceGL* srf = dynamic_cast<RHIInteropSurfaceGL*>(obj);
      if (srf) {
        GLuint textureID = srf->glId();
        RHIInteropSyncDescriptor desc = srf->syncDescriptor();

        if (desc.direction() & kSyncDirectionCUDAWriter) {
          m_waitTextureIDs.push_back(textureID);
          // TODO: Maybe support other layouts
          m_waitTextureSrcLayouts.push_back(GL_LAYOUT_GENERAL_EXT);
        }
        if (desc.direction() & kSyncDirectionRHIWriter) {
          m_signalTextureIDs.push_back(textureID);
          // TODO: Maybe support other layouts
          m_signalTextureDstLayouts.push_back(GL_LAYOUT_GENERAL_EXT);
        }
        continue;
      }
    }
    {
      // VK interop buffers carry no GL ID, so they don't go into any of the
      // lists above (those drive glSignal/WaitSemaphoreEXT, which doesn't
      // run in the no-GL stub mode anyway). Recognized here so the assert
      // below doesn't fire. Phase 6 replaces this whole class with
      // timeline-semaphore CUDA↔VK sync; until then we lose
      // producer/consumer ordering for VK-backed interop, same as the rest
      // of this transitional stub mode.
      if (dynamic_cast<RHIInteropBufferVK*>(obj))
        continue;
    }
    assert(false && "RHIInteropSync:rebuildSyncIDLists(): unhandled RHIObject type");
  }
}

void RHIInteropSync::signalCUDAToRHI(CUstream stream) {
  if (!m_hasGLContext) {
    // Stub mode: no GL consumer, so signaling would leave the semaphore in
    // a stuck signaled state. Just skip — the CUDA work still flushes via
    // the stream itself; we're losing CUDA↔VK ordering until Phase 6.
    return;
  }
  cudaExternalSemaphoreSignalParams params = {};
  CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&m_cudaToGlSemCU, &params, 1, stream));

  GL(glWaitSemaphoreEXT(m_cudaToGlSemGL,
    /*numBufferBarriers=*/ m_waitBufferIDs.size(), /*buffers=*/ m_waitBufferIDs.data(),
    /*numTextureBarriers=*/ m_waitTextureIDs.size(), /*textures=*/ m_waitTextureIDs.data(),
    /*srcLayouts=*/ m_waitTextureSrcLayouts.data()));
}

void RHIInteropSync::signalRHIToCUDA(CUstream stream) {
  if (!m_hasGLContext) {
    return;
  }
  GL(glSignalSemaphoreEXT(m_glToCudaSemGL,
    /*numBufferBarriers=*/ m_signalBufferIDs.size(), /*buffers=*/ m_signalBufferIDs.data(),
    /*numTextureBarriers=*/ m_signalTextureIDs.size(), /*textures=*/ m_signalTextureIDs.data(),
    /*dstLayouts=*/ m_signalTextureDstLayouts.data()));
  glFlush();

  cudaExternalSemaphoreWaitParams params = {};
  CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&m_glToCudaSemCU, &params, 1, stream));
}
