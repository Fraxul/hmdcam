#include "NvSciCudaInterop.h"
#include "common/tegra/NvSciUtil.h"
#include "rhi/cuda/CudaUtil.h"
#include "nvmedia_iofa.h"

#include <cstdio>
#include <cstdlib>
#include <cassert>

#define die(msg, ...) do { fprintf(stderr, msg"\n" , ##__VA_ARGS__); abort(); }while(0)

extern CUdevice cudaDevice;

NvSciSyncModule gSyncModule() {
  static NvSciSyncModule syncModule = nullptr;
  if (!syncModule) {
    NVSCI_CHECK(NvSciSyncModuleOpen(&syncModule));
  }
  return syncModule;
}

NvSciBufModule gBufModule() {
  static NvSciBufModule bufModule = nullptr;
  if (!bufModule) {
    NVSCI_CHECK(NvSciBufModuleOpen(&bufModule));
  }
  return bufModule;
}

NvSciCudaInteropBuffer::NvSciCudaInteropBuffer(NvSciBufAttrList attrList) {
  // Create NvSciBufObj
  NVSCI_CHECK(NvSciBufObjAlloc(attrList, &m_nvSciBuf));

  // Query some of the NvSciBuf attrs to set up CUDA interop
  NvSciBufAttrKeyValuePair queryAttrs[] {
    { NvSciBufImageAttrKey_Size, nullptr, 0 },
    { NvSciBufImageAttrKey_PlaneWidth, nullptr, 0 },
    { NvSciBufImageAttrKey_PlaneHeight, nullptr, 0 },
    { NvSciBufImageAttrKey_PlaneBitsPerPixel, nullptr, 0 },
    { NvSciBufImageAttrKey_PlaneDatatype, nullptr, 0 },
    { NvSciBufImageAttrKey_PlaneChannelCount, nullptr, 0 },
  };

  NVSCI_CHECK(NvSciBufAttrListGetAttrs(attrList, queryAttrs, sizeof(queryAttrs) / sizeof(queryAttrs[0])));

  m_bufferSizeBytes = *static_cast<const uint64_t*>(queryAttrs[0].value);
  m_width = *static_cast<const uint32_t*>(queryAttrs[1].value);
  m_height = *static_cast<const uint32_t*>(queryAttrs[2].value);
  m_bpp = *static_cast<const uint8_t*>(queryAttrs[3].value);
  NvSciBufAttrValDataType planeDataType = *static_cast<const NvSciBufAttrValDataType*>(queryAttrs[4].value);
  m_channelCount = *static_cast<const uint8_t*>(queryAttrs[5].value);

  //printf("NvSciCudaInteropBuffer: bufferSize=%lu plane=%ux%u bpp=%u dt=%u channels=%u\n",
  //  m_bufferSizeBytes, m_width, m_height, m_bpp, planeDataType, m_channelCount);

  // Import NvSciBufObj as CUDA external memory
  {
    CUDA_EXTERNAL_MEMORY_HANDLE_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF;
    desc.handle.nvSciBufObject = m_nvSciBuf;
    desc.size = m_bufferSizeBytes;
    CUDA_CHECK(cuImportExternalMemory(&m_cuMem, &desc));
  }

  // Map CUDA mipmapped array onto external memory
  {
    CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC desc;
    memset(&desc, 0, sizeof(desc));
    desc.offset = 0;
    desc.arrayDesc.Width = m_width;
    desc.arrayDesc.Height =  m_height;
    desc.arrayDesc.Depth = 0; // Depth=0 for 2D Array
    switch (planeDataType) {
      case NvSciDataType_Int8: desc.arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT8; break;
      case NvSciDataType_Uint8: desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8; break;
      case NvSciDataType_Int16: desc.arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT16; break;
      case NvSciDataType_Uint16: desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16; break;
      case NvSciDataType_Int32: desc.arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT32; break;
      case NvSciDataType_Uint32: desc.arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32; break;
      case NvSciDataType_Float16: desc.arrayDesc.Format = CU_AD_FORMAT_HALF; break;
      case NvSciDataType_Float32: desc.arrayDesc.Format = CU_AD_FORMAT_FLOAT; break;
      default: die("Unsupported NvSciBufAttrValDataType %u", planeDataType);
    }
    desc.arrayDesc.NumChannels = m_channelCount;
    desc.arrayDesc.Flags = 0;
    desc.numLevels = 1; // Fixed value for NvSciBuf imports

    CUDA_CHECK(cuExternalMemoryGetMappedMipmappedArray(&m_cuMipmappedArray, m_cuMem, &desc));
  }
  // Get CUarray handle to level 0 of the mipmapped array
  CUDA_CHECK(cuMipmappedArrayGetLevel(&m_cuArray, m_cuMipmappedArray, /*level=*/ 0));

}

NvSciCudaInteropBuffer::~NvSciCudaInteropBuffer() {
  CUDA_CHECK(cuMipmappedArrayDestroy(m_cuMipmappedArray));
  CUDA_CHECK(cuDestroyExternalMemory(m_cuMem));
  NvSciBufObjFree(m_nvSciBuf);
}

NvSciCudaInteropSync::NvSciCudaInteropSync(NvSciCudaInteropSyncDirection direction, NvMediaIofa* iofa) : m_direction(direction) {
  static NvSciSyncAttrList interopSyncAttrList[kNvSciCudaInteropSyncDirection_Count];

  // Demand-create the attribute list for this sync direction
  if (interopSyncAttrList[direction] == nullptr) {
    NvSciSyncAttrList syncAttrList[2];
    NVSCI_CHECK(NvSciSyncAttrListCreate(gSyncModule(), &syncAttrList[0]));
    NVSCI_CHECK(NvSciSyncAttrListCreate(gSyncModule(), &syncAttrList[1]));

    NVMEDIA_CHECK(NvMediaIOFAFillNvSciSyncAttrList(iofa, syncAttrList[0], direction == kSyncNvSciSignalerToCudaWaiter ? NVMEDIA_SIGNALER : NVMEDIA_WAITER));
    CUDA_CHECK(cuDeviceGetNvSciSyncAttributes(syncAttrList[1], cudaDevice, direction == kSyncNvSciSignalerToCudaWaiter ? CUDA_NVSCISYNC_ATTR_WAIT : CUDA_NVSCISYNC_ATTR_SIGNAL));

    NvSciSyncAttrList syncConflictList = nullptr;
    NVSCI_CHECK(NvSciSyncAttrListReconcile(syncAttrList, 2, &interopSyncAttrList[direction], &syncConflictList));

    assert(interopSyncAttrList[direction]);
    bool isReconciled = false;
    NVSCI_CHECK(NvSciSyncAttrListIsReconciled(interopSyncAttrList[direction], &isReconciled));
    assert(isReconciled);

    NvSciSyncAttrListFree(syncAttrList[0]);
    NvSciSyncAttrListFree(syncAttrList[1]);
  }

  NVSCI_CHECK(NvSciSyncObjAlloc(interopSyncAttrList[direction], &m_nvSciSync));
  CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc;
  memset(&desc, 0, sizeof(desc));
  desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC;
  desc.handle.nvSciSyncObj = m_nvSciSync;
  CUDA_CHECK(cuImportExternalSemaphore(&m_cuSem, &desc));
}

NvSciCudaInteropSync::~NvSciCudaInteropSync() {
  CUDA_CHECK(cuDestroyExternalSemaphore(m_cuSem));
  NvSciSyncObjFree(m_nvSciSync);
}

void NvSciCudaInteropSync::signalCudaToNvSci(CUstream hStream) {
  CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS params;
  memset(&params, 0, sizeof(params));
  params.params.nvSciSync.fence = &m_nvSciSyncFence;
  CUDA_CHECK(cuSignalExternalSemaphoresAsync(&m_cuSem, &params, /*numExtSems=*/ 1, /*stream=*/ hStream));
}

void NvSciCudaInteropSync::waitNvSciToCuda(CUstream hStream) {
  CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS params;
  memset(&params, 0, sizeof(params));
  params.params.nvSciSync.fence = &m_nvSciSyncFence;
  CUDA_CHECK(cuWaitExternalSemaphoresAsync(&m_cuSem, &params, /*numExtSems=*/ 1, /*stream=*/ hStream));
}

