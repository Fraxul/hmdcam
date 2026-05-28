#pragma once
#include <cuda.h>
#include "rhi/cuda/CudaUtil.h"

namespace RHICUDA {

void initRHICUDA();

extern bool initialized;

extern CUstream defaultAsyncStream;
extern CUdevice cudaDevice;
extern CUcontext cudaContext;

}; // namespace RHICUDA
