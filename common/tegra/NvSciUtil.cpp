#include "common/tegra/NvSciUtil.h"
#include "nvmedia_core.h"
#include "nvscierror.h"
#include <stdio.h>
#include <stdlib.h>

bool checkNvMediaStatus(NvMediaStatus res, const char* op, const char* file, int line, bool fatal) {
  if (res != NVMEDIA_STATUS_OK) {
    fprintf(stderr, "%s (%s:%d) returned NvMediaStatus %d\n", op, file, line, res);
    if (fatal)
      abort();

    return false;
  }
  return true;
}

bool checkNvSciError(NvSciError res, const char* op, const char* file, int line, bool fatal) {
  if (res != NvSciError_Success) {
    fprintf(stderr, "%s (%s:%d) returned NvSciError 0x%x\n", op, file, line, res);
    if (fatal)
      abort();

    return false;
  }
  return true;
}

NvSciSyncAttrList CreateNvSciSyncCpuSignalerAttrList(NvSciSyncModule syncModule) {
  NvSciSyncAttrList list = nullptr;
  NVSCI_CHECK(NvSciSyncAttrListCreate(syncModule, &list));

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
  return list;
}

NvSciSyncAttrList CreateNvSciSyncCpuWaiterAttrList(NvSciSyncModule syncModule) {
  NvSciSyncAttrList list = nullptr;
  NVSCI_CHECK(NvSciSyncAttrListCreate(syncModule, &list));

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
  return list;
}

const char* NvSciSyncAttrKey_toString(NvSciSyncAttrKey k) {
  static char defaultBuf[32];
  switch (k) {
    case NvSciSyncAttrKey_LowerBound: return "NvSciSyncAttrKey_LowerBound";
    case NvSciSyncAttrKey_NeedCpuAccess: return "NvSciSyncAttrKey_NeedCpuAccess";
    case NvSciSyncAttrKey_RequiredPerm: return "NvSciSyncAttrKey_RequiredPerm";
    case NvSciSyncAttrKey_ActualPerm: return "NvSciSyncAttrKey_ActualPerm";
    case NvSciSyncAttrKey_WaiterContextInsensitiveFenceExports: return "NvSciSyncAttrKey_WaiterContextInsensitiveFenceExports";
    case NvSciSyncAttrKey_WaiterRequireTimestamps: return "NvSciSyncAttrKey_WaiterRequireTimestamps";
    case NvSciSyncAttrKey_RequireDeterministicFences: return "NvSciSyncAttrKey_RequireDeterministicFences";
    case NvSciSyncAttrKey_UpperBound: return "NvSciSyncAttrKey_UpperBound";

    default:
      snprintf(defaultBuf, 32, "0x%x", k);
      return defaultBuf;
  }
}

void DumpNvSciSyncAttrList(NvSciSyncAttrList list) {
  for (uint32_t key = NvSciSyncAttrKey_LowerBound; key < NvSciSyncAttrKey_UpperBound; ++key) {
    const void* buf = nullptr;
    size_t len = 0;
    if (NvSciSyncAttrListGetAttr(list, (NvSciSyncAttrKey) key, &buf, &len) == NvSciError_Success && buf != nullptr && len != 0) {
      fprintf(stderr, "  %s: ", NvSciSyncAttrKey_toString((NvSciSyncAttrKey) key));
      for (size_t i = 0; i < len; ++i) {
        fprintf(stderr, "%02x ", reinterpret_cast<const char*>(buf)[i]);
      }
      fprintf(stderr, "\n");
    }
  }
}

NvSciSyncAttrList ReconcileNvSciSyncAttrLists(NvSciSyncAttrList list1, NvSciSyncAttrList list2) {
  NvSciSyncAttrList reconciledList = nullptr, conflictList = nullptr;
  NvSciSyncAttrList lists[2];
  lists[0] = list1;
  lists[1] = list2;

  NvSciError err = NvSciSyncAttrListReconcile(lists, (list2 == nullptr) ? 1 : 2, &reconciledList, &conflictList);

  if (err == NvSciError_ReconciliationFailed) {
    fprintf(stderr, "ReconcileNvSciSyncAttrLists reconciliation failed.\n");
    fprintf(stderr, "list1:\n");
    DumpNvSciSyncAttrList(list1);

    if (list2 != nullptr) {
      fprintf(stderr, "list2:\n");
      DumpNvSciSyncAttrList(list2);
    }

    fprintf(stderr, "Conflicting attributes:\n");
    DumpNvSciSyncAttrList(conflictList);

    abort();
  } else if (err != NvSciError_Success) {
    fprintf(stderr, "ReconcileNvSciSyncAttrLists reconciliation failed with NvSciError 0x%x\n", err);
    abort();
  }
  NvSciSyncAttrListFree(list1);
  if (list2 != nullptr)
    NvSciSyncAttrListFree(list2);

  return reconciledList;
}
