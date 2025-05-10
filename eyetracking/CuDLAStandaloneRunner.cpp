/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "CuDLAStandaloneRunner.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include "common/mmfile.h"
#include "common/tegra/NvSciUtil.h"
#include <unistd.h>

#define CleanupPtr(Fn, Obj, ... ) if (Obj != nullptr) { Fn(Obj  ,## __VA_ARGS__); Obj = nullptr; }

#define CUDLA_CHECK(x) checkCUDLAstatus(x, #x, __FILE__, __LINE__, true)
#define CUDLA_CHECK_NONFATAL(x) checkCUDLAstatus(x, #x, __FILE__, __LINE__, false)
bool checkCUDLAstatus(cudlaStatus st, const char* op, const char* file, int line, bool fatal) {
  if (st != cudlaSuccess) {
    fprintf(stderr, "%s (%s:%d) returned cudlaStatus %d\n", op, file, line, st);
    if (fatal)
      abort();
    return false;
  }
  return true;
}

static void printTensorDesc(const std::vector<cudlaModuleTensorDescriptor>& tensorDescs) {
  for (size_t idx = 0; idx < tensorDescs.size(); ++idx) {
    const cudlaModuleTensorDescriptor& desc = tensorDescs[idx];

    printf("\tTENSOR %zu NAME : %s\n", idx, desc.name);
    printf("\tsize: %lu\n", desc.size);

    printf("\tdims: [%lu, %lu, %lu, %lu]\n", desc.n, desc.c, desc.h, desc.w);

    printf("\tdata fmt: %d\n", desc.dataFormat);
    printf("\tdata type: %d\n", desc.dataType);
    printf("\tdata category: %d\n", desc.dataCategory);
    printf("\tpixel fmt: %d\n", desc.pixelFormat);
    printf("\tpixel mapping: %d\n", desc.pixelMapping);
    printf("\tstride[0]: %d\n", desc.stride[0]);
    printf("\tstride[1]: %d\n", desc.stride[1]);
    printf("\tstride[2]: %d\n", desc.stride[2]);
    printf("\tstride[3]: %d\n", desc.stride[3]);
  }
}


CuDLAStandaloneRunner::~CuDLAStandaloneRunner() {
  if (m_inputBufObjRegPtr)
    CUDLA_CHECK_NONFATAL(cudlaMemUnregister(m_devHandle, m_inputBufObjRegPtr));
  if (m_outputBufObjRegPtr)
    CUDLA_CHECK_NONFATAL(cudlaMemUnregister(m_devHandle, m_outputBufObjRegPtr));

  CleanupPtr(NvSciBufObjFree, m_inputBufObj);
  CleanupPtr(NvSciBufObjFree, m_outputBufObj);
  CleanupPtr(NvSciBufAttrListFree, m_reconciledInputAttrList);
  CleanupPtr(NvSciBufAttrListFree, m_inputConflictList);
  CleanupPtr(NvSciBufAttrListFree, m_inputAttrList);
  CleanupPtr(NvSciBufAttrListFree, m_reconciledOutputAttrList);
  CleanupPtr(NvSciBufAttrListFree, m_outputConflictList);
  CleanupPtr(NvSciBufAttrListFree, m_outputAttrList);
  CleanupPtr(NvSciBufModuleClose, m_bufModule);

  NvSciSyncFenceClear(&(m_preFence));
  NvSciSyncFenceClear(&(m_eofFence));

  CleanupPtr(NvSciSyncObjFree, m_syncObj1);
  CleanupPtr(NvSciSyncObjFree, m_syncObj2);
  CleanupPtr(NvSciSyncAttrListFree, m_nvSciSyncConflictListObj1);
  CleanupPtr(NvSciSyncAttrListFree, m_nvSciSyncReconciledListObj1);
  CleanupPtr(NvSciSyncAttrListFree, m_nvSciSyncConflictListObj2);
  CleanupPtr(NvSciSyncAttrListFree, m_nvSciSyncReconciledListObj2);
  CleanupPtr(NvSciSyncAttrListFree, m_signalerAttrListObj1);
  CleanupPtr(NvSciSyncAttrListFree, m_waiterAttrListObj1);
  CleanupPtr(NvSciSyncAttrListFree, m_signalerAttrListObj2);
  CleanupPtr(NvSciSyncAttrListFree, m_waiterAttrListObj2);
  CleanupPtr(NvSciSyncCpuWaitContextFree, m_nvSciCtx);
  CleanupPtr(NvSciSyncModuleClose, m_syncModule);

  if (m_nvSciSyncObjRegPtr1)
    cudlaMemUnregister(m_devHandle, m_nvSciSyncObjRegPtr1);

  if (m_nvSciSyncObjRegPtr2)
    cudlaMemUnregister(m_devHandle, m_nvSciSyncObjRegPtr2);

  CleanupPtr(cudlaModuleUnload, m_moduleHandle, 0);
  CleanupPtr(cudlaDestroyDevice, m_devHandle);
}

void createAndSetAttrList(NvSciBufModule module, uint64_t bufSize, NvSciBufAttrList *attrList)
{
  NVSCI_CHECK(NvSciBufAttrListCreate(module, attrList));

  bool                      needCpuAccess = true;
  NvSciBufAttrValAccessPerm perm          = NvSciBufAccessPerm_ReadWrite;
  uint32_t                  dimcount      = 1;
  uint64_t                  sizes[]       = {bufSize};
  uint32_t                  alignment[]   = {1};
  uint32_t                  dataType      = NvSciDataType_Int8;
  NvSciBufType              type          = NvSciBufType_Tensor;
  uint64_t                  baseAddrAlign = 512;

  NvSciBufAttrKeyValuePair setAttrs[] = {
      {.key = NvSciBufGeneralAttrKey_Types, .value = &type, .len = sizeof(type)},
      {.key = NvSciBufTensorAttrKey_DataType, .value = &dataType, .len = sizeof(dataType)},
      {.key = NvSciBufTensorAttrKey_NumDims, .value = &dimcount, .len = sizeof(dimcount)},
      {.key = NvSciBufTensorAttrKey_SizePerDim, .value = &sizes, .len = sizeof(sizes)},
      {.key = NvSciBufTensorAttrKey_AlignmentPerDim, .value = &alignment, .len = sizeof(alignment)},
      {.key = NvSciBufTensorAttrKey_BaseAddrAlign, .value = &baseAddrAlign, .len = sizeof(baseAddrAlign)},
      {.key = NvSciBufGeneralAttrKey_RequiredPerm, .value = &perm, .len = sizeof(perm)},
      {.key = NvSciBufGeneralAttrKey_NeedCpuAccess, .value = &needCpuAccess, .len = sizeof(needCpuAccess)}};
  size_t length = sizeof(setAttrs) / sizeof(NvSciBufAttrKeyValuePair);

  NVSCI_CHECK(NvSciBufAttrListSetAttrs(*attrList, setAttrs, length));
}

void fillCpuSignalerAttrList(NvSciSyncAttrList list)
{
  bool                      cpuSignaler = true;
  NvSciSyncAttrKeyValuePair keyValue[2];
  memset(keyValue, 0, sizeof(keyValue));
  keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
  keyValue[0].value   = (void *)&cpuSignaler;
  keyValue[0].len     = sizeof(cpuSignaler);

  NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_SignalOnly;
  keyValue[1].attrKey         = NvSciSyncAttrKey_RequiredPerm;
  keyValue[1].value           = (void *)&cpuPerm;
  keyValue[1].len             = sizeof(cpuPerm);

  NVSCI_CHECK(NvSciSyncAttrListSetAttrs(list, keyValue, 2));
}

void fillCpuWaiterAttrList(NvSciSyncAttrList list) {
  bool                      cpuWaiter = true;
  NvSciSyncAttrKeyValuePair keyValue[2];
  memset(keyValue, 0, sizeof(keyValue));
  keyValue[0].attrKey = NvSciSyncAttrKey_NeedCpuAccess;
  keyValue[0].value   = (void *)&cpuWaiter;
  keyValue[0].len     = sizeof(cpuWaiter);

  NvSciSyncAccessPerm cpuPerm = NvSciSyncAccessPerm_WaitOnly;
  keyValue[1].attrKey         = NvSciSyncAttrKey_RequiredPerm;
  keyValue[1].value           = (void *)&cpuPerm;
  keyValue[1].len             = sizeof(cpuPerm);

  NVSCI_CHECK(NvSciSyncAttrListSetAttrs(list, keyValue, 2));
}

CuDLAStandaloneRunner::CuDLAStandaloneRunner(uint64_t deviceIdx, const char* engineFile) {
  mmfile fp(engineFile);

  CUDLA_CHECK(cudlaCreateDevice(deviceIdx, &m_devHandle, CUDLA_STANDALONE));
  CUDLA_CHECK(cudlaModuleLoadFromMemory(m_devHandle, reinterpret_cast<const uint8_t*>(fp.data()), fp.size(), &m_moduleHandle, 0));

  // Get tensor attributes.
  cudlaModuleAttribute attribute;

  CUDLA_CHECK(cudlaModuleGetAttributes(m_moduleHandle, CUDLA_NUM_INPUT_TENSORS, &attribute));

  uint32_t numInputTensors = attribute.numInputTensors;
  printf("numInputTensors = %d\n", numInputTensors);

  CUDLA_CHECK(cudlaModuleGetAttributes(m_moduleHandle, CUDLA_NUM_OUTPUT_TENSORS, &attribute));
  uint32_t numOutputTensors = attribute.numOutputTensors;
  printf("numOutputTensors = %d\n", numOutputTensors);

  m_inputTensorDesc.resize(numInputTensors);
  m_outputTensorDesc.resize(numOutputTensors);

  attribute.inputTensorDesc = m_inputTensorDesc.data();
  CUDLA_CHECK(cudlaModuleGetAttributes(m_moduleHandle, CUDLA_INPUT_TENSOR_DESCRIPTORS, &attribute));

  printf("Printing input tensor descriptor\n");
  printTensorDesc(m_inputTensorDesc);

  attribute.outputTensorDesc = m_outputTensorDesc.data();
  CUDLA_CHECK(cudlaModuleGetAttributes(m_moduleHandle, CUDLA_OUTPUT_TENSOR_DESCRIPTORS, &attribute));

  printf("Printing output tensor descriptor\n");
  printTensorDesc(m_outputTensorDesc);

  NVSCI_CHECK(NvSciBufModuleOpen(&m_bufModule));

  // creating and setting input attribute list
  createAndSetAttrList(m_bufModule, m_inputTensorDesc[0].size, &m_inputAttrList);


  NVSCI_CHECK(NvSciBufAttrListReconcile(&m_inputAttrList, 1, &m_reconciledInputAttrList, &m_inputConflictList));

  // creating and setting output attribute list
  createAndSetAttrList(m_bufModule, m_outputTensorDesc[0].size, &m_outputAttrList);


  NVSCI_CHECK(NvSciBufAttrListReconcile(&m_outputAttrList, 1, &m_reconciledOutputAttrList, &m_outputConflictList));

  NVSCI_CHECK(NvSciBufObjAlloc(m_reconciledInputAttrList, &m_inputBufObj));

  NVSCI_CHECK(NvSciBufObjAlloc(m_reconciledOutputAttrList, &m_outputBufObj));

  // importing external memory
  cudlaExternalMemoryHandleDesc memDesc = {0};
  memset(&memDesc, 0, sizeof(memDesc));
  memDesc.extBufObject = (void *)m_inputBufObj;
  memDesc.size         = m_inputTensorDesc[0].size;
  CUDLA_CHECK(cudlaImportExternalMemory(m_devHandle, &memDesc, &m_inputBufObjRegPtr, 0));

  NVSCI_CHECK(NvSciBufObjGetCpuPtr(m_inputBufObj, &m_inputBufObjBuffer));

  memset(&memDesc, 0, sizeof(memDesc));
  memDesc.extBufObject = (void *)m_outputBufObj;
  memDesc.size         = m_outputTensorDesc[0].size;
  CUDLA_CHECK(cudlaImportExternalMemory(m_devHandle, &memDesc, &m_outputBufObjRegPtr, 0));

  NVSCI_CHECK(NvSciBufObjGetCpuPtr(m_outputBufObj, &m_outputBufObjBuffer));

  NVSCI_CHECK(NvSciSyncModuleOpen(&m_syncModule));
  NVSCI_CHECK(NvSciSyncAttrListCreate(m_syncModule, &m_signalerAttrListObj1));
  NVSCI_CHECK(NvSciSyncAttrListCreate(m_syncModule, &m_waiterAttrListObj1));
  CUDLA_CHECK(cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t *>(m_waiterAttrListObj1), CUDLA_NVSCISYNC_ATTR_WAIT));
  fillCpuSignalerAttrList(m_signalerAttrListObj1);

  {
    NvSciSyncAttrList  syncAttrListObj1[2];
    syncAttrListObj1[0] = m_signalerAttrListObj1;
    syncAttrListObj1[1] = m_waiterAttrListObj1;
    NVSCI_CHECK(NvSciSyncAttrListReconcile(syncAttrListObj1, 2, &m_nvSciSyncReconciledListObj1, &m_nvSciSyncConflictListObj1));
  }

  NVSCI_CHECK(NvSciSyncObjAlloc(m_nvSciSyncReconciledListObj1, &m_syncObj1));

  NVSCI_CHECK(NvSciSyncCpuWaitContextAlloc(m_syncModule, &m_nvSciCtx));

  NVSCI_CHECK(NvSciSyncAttrListCreate(m_syncModule, &m_signalerAttrListObj2));

  NVSCI_CHECK(NvSciSyncAttrListCreate(m_syncModule, &m_waiterAttrListObj2));

  CUDLA_CHECK(cudlaGetNvSciSyncAttributes(reinterpret_cast<uint64_t *>(m_signalerAttrListObj2), CUDLA_NVSCISYNC_ATTR_SIGNAL));

  fillCpuWaiterAttrList(m_waiterAttrListObj2);

  {
    NvSciSyncAttrList syncAttrListObj2[2];
    syncAttrListObj2[0] = m_signalerAttrListObj2;
    syncAttrListObj2[1] = m_waiterAttrListObj2;
    NVSCI_CHECK(NvSciSyncAttrListReconcile(syncAttrListObj2, 2, &m_nvSciSyncReconciledListObj2, &m_nvSciSyncConflictListObj2));
  }

  NVSCI_CHECK(NvSciSyncObjAlloc(m_nvSciSyncReconciledListObj2, &m_syncObj2));

  // importing external semaphore
  cudlaExternalSemaphoreHandleDesc semaMemDesc         = {0};
  memset(&semaMemDesc, 0, sizeof(semaMemDesc));
  semaMemDesc.extSyncObject = m_syncObj1;
  CUDLA_CHECK(cudlaImportExternalSemaphore(m_devHandle, &semaMemDesc, &m_nvSciSyncObjRegPtr1, 0));

  memset(&semaMemDesc, 0, sizeof(semaMemDesc));
  semaMemDesc.extSyncObject = m_syncObj2;
  CUDLA_CHECK(cudlaImportExternalSemaphore(m_devHandle, &semaMemDesc, &m_nvSciSyncObjRegPtr2, 0));

  // Wait events
  m_preFence = NvSciSyncFenceInitializer;
  NVSCI_CHECK(NvSciSyncObjGenerateFence(m_syncObj1, &m_preFence));

  memset(&m_waitEvents, 0, sizeof(m_waitEvents));
  m_waitEvents.numEvents = 1;

  memset(m_preFences, 0, sizeof(m_preFences));

  m_preFences[0].fence      = &m_preFence;
  m_preFences[0].type       = CUDLA_NVSCISYNC_FENCE;

  m_waitEvents.preFences = m_preFences;

  // Signal Events
  memset(m_signalEventDevPtrs, 0, sizeof(m_signalEventDevPtrs));

  m_signalEventDevPtrs[0]            = m_nvSciSyncObjRegPtr2;

  memset(&m_signalEvents, 0, sizeof(m_signalEvents));
  m_signalEvents.numEvents = 1;
  m_signalEvents.devPtrs = m_signalEventDevPtrs;
  m_signalEvents.eofFences = m_eofFences;

  m_eofFence = NvSciSyncFenceInitializer;
  m_eofFences[0].fence = &m_eofFence;
  m_eofFences[0].type  = CUDLA_NVSCISYNC_FENCE;


  // Setup task struct, since it'll always be the same
  m_task.moduleHandle     = m_moduleHandle;
  m_task.outputTensor     = &m_outputBufObjRegPtr;
  m_task.numOutputTensors = 1;
  m_task.numInputTensors  = 1;
  m_task.inputTensor      = &m_inputBufObjRegPtr;
  m_task.waitEvents       = &m_waitEvents;
  m_task.signalEvents     = &m_signalEvents;
}


void CuDLAStandaloneRunner::runInference() {
    // Enqueue a cuDLA task.
    CUDLA_CHECK(cudlaSubmitTask(m_devHandle, &m_task, 1, NULL, 0));

    // XXX copy input data to inputBufObjBuffer
    // memcpy(inputBufObjBuffer, inputBuffer, inputTensorDesc[0].size);

    // Signal wait events
    NvSciSyncObjSignal(m_syncObj1);

    // Wait for operations to finish and bring output buffer to CPU.
    NVSCI_CHECK(NvSciSyncFenceWait(reinterpret_cast<NvSciSyncFence *>(m_signalEvents.eofFences[0].fence), m_nvSciCtx, -1));

    // Output is available in outputBufObjBuffer.
}

