#ifdef HAVE_VPI2
#include "common/VPIUtil.h"
#include <stdio.h>
#include <stdlib.h>

bool checkVPIStatus(VPIStatus res, const char* op, const char* file, int line, bool fatal) {
  if (res != VPI_SUCCESS) {
    fprintf(stderr, "%s (%s:%d) returned VPIStatus %d: %s\n", op, file, line, res, vpiStatusGetName(res));
    if (fatal)
      abort();
    return false;
  }
  return true;
}

#endif // HAVE_VPI2
