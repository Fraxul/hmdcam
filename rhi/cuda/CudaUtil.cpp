#include <cuda.h>
#include <stdio.h>
#include "rhi/cuda/CudaUtil.h"

bool checkCUresult(CUresult res, const char* op, const char* file, int line, bool fatal) {
  if (res != CUDA_SUCCESS) {
    const char* errorDesc = NULL;
    cuGetErrorString(res, &errorDesc);
    fprintf(stderr, "%s (%s:%d) returned CUresult %d: %s\n", op, file, line, res, errorDesc);
    if (fatal)
      abort();
    return false;
  }
  return true;
}

