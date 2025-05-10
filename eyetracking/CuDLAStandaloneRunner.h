#pragma once

#include <boost/noncopyable.hpp>
#include <vector>
#include "cudla.h"
#include "nvscibuf.h"
#include "nvscierror.h"
#include "nvscisync.h"



class CuDLAStandaloneRunner : boost::noncopyable {
public:
  CuDLAStandaloneRunner(uint64_t deviceIdx, const char* engineFile);
  ~CuDLAStandaloneRunner();

  void runInference();

protected:
  cudlaDevHandle               m_devHandle = nullptr;
  cudlaModule                  m_moduleHandle = nullptr;
  NvSciBufObj                  m_inputBufObj = nullptr;
  NvSciBufObj                  m_outputBufObj = nullptr;
  NvSciBufModule               m_bufModule = nullptr;
  NvSciBufAttrList             m_inputAttrList = nullptr;
  NvSciBufAttrList             m_reconciledInputAttrList = nullptr;
  NvSciBufAttrList             m_inputConflictList = nullptr;
  NvSciBufAttrList             m_outputAttrList = nullptr;
  NvSciBufAttrList             m_reconciledOutputAttrList = nullptr;
  NvSciBufAttrList             m_outputConflictList = nullptr;
  NvSciSyncObj                 m_syncObj1 = nullptr;
  NvSciSyncObj                 m_syncObj2 = nullptr;
  NvSciSyncModule              m_syncModule = nullptr;
  NvSciSyncFence               m_preFence = NvSciSyncFenceInitializer;
  NvSciSyncFence               m_eofFence = NvSciSyncFenceInitializer;
  NvSciSyncCpuWaitContext      m_nvSciCtx = nullptr;
  NvSciSyncAttrList            m_waiterAttrListObj1 = nullptr;
  NvSciSyncAttrList            m_signalerAttrListObj1 = nullptr;
  NvSciSyncAttrList            m_waiterAttrListObj2 = nullptr;
  NvSciSyncAttrList            m_signalerAttrListObj2 = nullptr;
  NvSciSyncAttrList            m_nvSciSyncConflictListObj1 = nullptr;
  NvSciSyncAttrList            m_nvSciSyncReconciledListObj1 = nullptr;
  NvSciSyncAttrList            m_nvSciSyncConflictListObj2 = nullptr;
  NvSciSyncAttrList            m_nvSciSyncReconciledListObj2 = nullptr;
  std::vector<cudlaModuleTensorDescriptor> m_inputTensorDesc;
  std::vector<cudlaModuleTensorDescriptor> m_outputTensorDesc;

  cudlaWaitEvents             m_waitEvents;
  cudlaSignalEvents           m_signalEvents;
  CudlaFence                  m_preFences[1];
  CudlaFence                  m_eofFences[1];
  uint64_t* m_signalEventDevPtrs[1];
  uint64_t* m_inputBufObjRegPtr = nullptr;
  uint64_t* m_outputBufObjRegPtr = nullptr;
  uint64_t* m_nvSciSyncObjRegPtr1 = nullptr;
  uint64_t* m_nvSciSyncObjRegPtr2 = nullptr;

  // CPU-accessible I/O buffer pointers
  void* m_inputBufObjBuffer = nullptr;
  void* m_outputBufObjBuffer = nullptr;

  // Task struct
  cudlaTask m_task;
};
