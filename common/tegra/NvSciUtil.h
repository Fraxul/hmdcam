#pragma once
#include "nvmedia_core.h"
#include "nvscierror.h"
#include "nvscisync.h"
#include "nvscibuf.h"

#define NVMEDIA_CHECK(x) checkNvMediaStatus(x, #x, __FILE__, __LINE__, true)
#define NVSCI_CHECK(x) checkNvSciError(x, #x, __FILE__, __LINE__, true)
bool checkNvMediaStatus(NvMediaStatus res, const char* op, const char* file, int line, bool fatal);
bool checkNvSciError(NvSciError res, const char* op, const char* file, int line, bool fatal);

NvSciSyncAttrList CreateNvSciSyncCpuSignalerAttrList(NvSciSyncModule syncModule);
NvSciSyncAttrList CreateNvSciSyncCpuWaiterAttrList(NvSciSyncModule syncModule);

// Reconciles one or two lists together. Frees input attribute list(s).
NvSciSyncAttrList ReconcileNvSciSyncAttrLists(NvSciSyncAttrList list1, NvSciSyncAttrList list2 = nullptr);
void DumpNvSciSyncAttrList(NvSciSyncAttrList list);

NvSciBufAttrList ReconcileNvSciBufAttrLists(NvSciBufAttrList list1, NvSciBufAttrList list2 = nullptr);
void DumpNvSciBufAttrList(NvSciBufAttrList list);

