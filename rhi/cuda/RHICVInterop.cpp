#include "rhi/cuda/RHICVInterop.h"
#include "rhi/CudaUtil.h"
#include <cuda.h>

namespace RHICUDA {

void copyGpuMatToSurface(const cv::cuda::GpuMat& gpuMat, RHISurface::ptr surface, CUstream stream) {

  size_t copyWidth = std::min<size_t>(surface->width(), gpuMat.cols);
  size_t copyHeight = std::min<size_t>(surface->height(), gpuMat.rows);

  CUarray pSurfaceMip0Array;
  CUDA_CHECK(cuGraphicsResourceSetMapFlags(surface->cuGraphicsResource(), CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD));
  CUDA_CHECK(cuGraphicsMapResources(1, &surface->cuGraphicsResource(), /*CUstream*/ 0));
  CUDA_CHECK(cuGraphicsSubResourceGetMappedArray(&pSurfaceMip0Array, surface->cuGraphicsResource(), 0, 0));

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.srcDevice = (CUdeviceptr) gpuMat.cudaPtr();
  copyDescriptor.srcPitch = gpuMat.step;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.dstArray = pSurfaceMip0Array;

  copyDescriptor.WidthInBytes = copyWidth * gpuMat.elemSize();
  copyDescriptor.Height = copyHeight;
  CUDA_CHECK(cuMemcpy2D(&copyDescriptor));

  CUDA_CHECK(cuGraphicsUnmapResources(1, &surface->cuGraphicsResource(), /*CUstream*/ 0));
}


};
