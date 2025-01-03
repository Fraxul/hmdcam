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

