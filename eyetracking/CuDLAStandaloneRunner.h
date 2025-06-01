#pragma once

#include <boost/noncopyable.hpp>
#include <vector>
#include <cassert>
#include "cudla.h"
#include "nvscibuf.h"
#include "nvscierror.h"
#include "nvscisync.h"

class CuDLAStandaloneRunner : boost::noncopyable {
public:
  CuDLAStandaloneRunner(uint64_t deviceIdx, const char* engineFile);
  CuDLAStandaloneRunner(uint64_t deviceIdx, const uint8_t* moduleData, size_t moduleLen);

  ~CuDLAStandaloneRunner();

  void runInference() {
    asyncStartInference();
    asyncFinishInference();
  }

  void asyncStartInference();
  void asyncFinishInference();

/*
typedef struct cudlaModuleTensorDescriptor_t {
    char name[CUDLA_RUNTIME_TENSOR_DESC_NAME_MAX_LEN + 1];
    uint64_t size; // full size in bytes
    uint64_t n;
    uint64_t c;
    uint64_t h;
    uint64_t w;
    uint8_t dataFormat;   // CUDLA_DATA_FORMAT_*
    uint8_t dataType;     // CUDLA_DATA_TYPE_*
    uint8_t dataCategory; // CUDLA_DATA_CATEGORY_*
    uint8_t pixelFormat;  // CUDLA_PIXEL_FORMAT_*
    uint8_t pixelMapping; // CUDLA_PIXEL_MAPPING_{PITCH|BLOCK}_LINEAR
    uint32_t stride[CUDLA_LOADABLE_TENSOR_DESC_NUM_STRIDES]; // w, h, c, n -- stride in bytes
} cudlaModuleTensorDescriptor;
*/

  size_t inputTensorCount() const { return m_inputTensorDesc.size(); }
  const cudlaModuleTensorDescriptor& inputTensorDescriptor(size_t idx) const { return m_inputTensorDesc[idx]; }
  template <typename T = void> T* inputTensorPtr(size_t idx) const { assert(idx == 0); return reinterpret_cast<T*>(m_inputBufObjBuffer); }

  size_t outputTensorCount() const { return m_outputTensorDesc.size(); }
  const cudlaModuleTensorDescriptor& outputTensorDescriptor(size_t idx) const { return m_outputTensorDesc[idx]; }
  template <typename T = void> T* outputTensorPtr(size_t idx) const { assert(idx == 0); return reinterpret_cast<T*>(m_outputBufObjBuffer); }

protected:
  void initWithModuleData(uint64_t deviceIdx, const uint8_t* moduleData, size_t moduleLen);

  cudlaDevHandle               m_devHandle = nullptr;
  cudlaModule                  m_moduleHandle = nullptr;
  NvSciBufObj                  m_inputBufObj = nullptr;
  NvSciBufObj                  m_outputBufObj = nullptr;
  NvSciBufModule               m_bufModule = nullptr;
  NvSciSyncObj                 m_syncWaitObj = nullptr;
  NvSciSyncObj                 m_syncSignalObj = nullptr;
  NvSciSyncModule              m_syncModule = nullptr;
  NvSciSyncFence               m_preFence = NvSciSyncFenceInitializer; // Associated with m_syncWaitObj
  NvSciSyncFence               m_eofFence = NvSciSyncFenceInitializer; // Associated with m_syncSignalObj
  NvSciSyncCpuWaitContext      m_cpuWaitCtx = nullptr;
  std::vector<cudlaModuleTensorDescriptor> m_inputTensorDesc;
  std::vector<cudlaModuleTensorDescriptor> m_outputTensorDesc;

  cudlaWaitEvents             m_waitEvents;
  cudlaSignalEvents           m_signalEvents;
  CudlaFence                  m_preFences[1];
  CudlaFence                  m_eofFences[1];

  // Deterministic fence support
  uint64_t m_signalerID = 0;
  uint64_t m_signalerValue = 0;
  uint64_t m_waiterID = 0;
  uint64_t m_waiterValue = 0;

  uint64_t* m_signalEventDevPtrs[1];
  uint64_t* m_inputBufObjRegPtr = nullptr;
  uint64_t* m_outputBufObjRegPtr = nullptr;
  uint64_t* m_syncWaitObjRegPtr = nullptr;
  uint64_t* m_syncSignalObjRegPtr = nullptr;

  // CPU-accessible I/O buffer pointers
  void* m_inputBufObjBuffer = nullptr;
  void* m_outputBufObjBuffer = nullptr;

  // Task struct
  cudlaTask m_task;
};
