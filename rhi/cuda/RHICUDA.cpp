#include "rhi/cuda/RHICUDA.h"
#include <cstdio>

namespace RHICUDA {

CUstream defaultAsyncStream;
CUdevice cudaDevice;
CUcontext cudaContext;

void initRHICUDA() {
  cuInit(0);

  cuDeviceGet(&cudaDevice, 0);
  char devName[512];
  cuDeviceGetName(devName, 511, cudaDevice);
  devName[511] = '\0';
  printf("CUDA device: %s\n", devName);


  // Explicit driver API init with the runtime's primary context.
  // This ensures that the driver and runtime APIs will use the same context.
  cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice);
  cuCtxSetCurrent(cudaContext);

  // Create a default non-blocking stream.
  CUDA_CHECK(cuStreamCreate(&defaultAsyncStream, CU_STREAM_NON_BLOCKING));
}

}; // namespace RHICUDA
