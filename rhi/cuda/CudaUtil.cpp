#include <cuda.h>
#include <stdio.h>
#include "rhi/cuda/CudaUtil.h"

void checkCUresult(CUresult res, const char* op, const char* file, int line) {
  if (res != CUDA_SUCCESS) {
    const char* errorDesc = NULL;
    cuGetErrorString(res, &errorDesc);
    fprintf(stderr, "%s (%s:%d) returned CUresult %d: %s\n", op, file, line, res, errorDesc);
    abort();
  }
}

