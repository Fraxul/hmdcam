#pragma once
#include <boost/core/noncopyable.hpp>
#include <cuda.h>
#include "nvscibuf.h"
#include "nvscisync.h"

// Global NvSci module instances, created on-demand
extern NvSciSyncModule gSyncModule();
extern NvSciBufModule gBufModule();


class NvSciCudaInteropBuffer : boost::noncopyable {
public:
  NvSciCudaInteropBuffer(NvSciBufAttrList attrList);
  ~NvSciCudaInteropBuffer();

  NvSciBufObj m_nvSciBuf = nullptr;
  CUexternalMemory m_cuMem;
  CUmipmappedArray m_cuMipmappedArray;
  CUarray m_cuArray;

  uint64_t m_bufferSizeBytes;
  uint32_t m_width, m_height;
  uint8_t m_bpp, m_channelCount;
};

class NvSciCudaInteropSync : boost::noncopyable {
public:
  enum NvSciCudaInteropSyncDirection {
    kSyncNvSciSignalerToCudaWaiter,
    kSyncCudaSignalerToNvSciWaiter,

    kNvSciCudaInteropSyncDirection_Count
  };

  NvSciCudaInteropSync(NvSciCudaInteropSyncDirection direction);
  ~NvSciCudaInteropSync();

  void signalCudaToNvSci(CUstream hStream);
  void waitNvSciToCuda(CUstream hStream);

  NvSciCudaInteropSyncDirection m_direction;
  NvSciSyncObj m_nvSciSync = nullptr;
  NvSciSyncFence m_nvSciSyncFence = NvSciSyncFenceInitializer;
  CUexternalSemaphore m_cuSem;
};

