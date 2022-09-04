#include <cuda.h>
#include <cuda_runtime.h>
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

bool checkCUresult(cudaError_t res, const char* op, const char* file, int line, bool fatal) {
  if (res != cudaSuccess) {
    fprintf(stderr, "%s (%s:%d): CUDA error %d: %s: %s\n", op, file, line, res, cudaGetErrorName(res), cudaGetErrorString(res));
    if (fatal)
      abort();
    return false;
  }
  return true;
}
