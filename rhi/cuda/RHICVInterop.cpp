#include "rhi/cuda/RHICVInterop.h"
#ifdef HAVE_CUDA
#include "rhi/cuda/CudaUtil.h"
#include <cuda.h>

namespace RHICUDA {

void copyGpuMatToSurface(const cv::cuda::GpuMat& gpuMat, RHISurface::ptr surface, const cv::cuda::Stream& stream) {
  copyGpuMatToSurface(gpuMat, surface, (CUstream) stream.cudaPtr());
}

void copyGpuMatToSurface(const cv::cuda::GpuMat& gpuMat, RHISurface::ptr surface, CUstream stream) {
  assert(surface->isInteropSurface());

  size_t copyWidth = std::min<size_t>(surface->width(), gpuMat.cols);
  size_t copyHeight = std::min<size_t>(surface->height(), gpuMat.rows);

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.srcDevice = (CUdeviceptr) gpuMat.cudaPtr();
  copyDescriptor.srcPitch = gpuMat.step;

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.dstArray = (CUarray) surface->cudaArray();

  copyDescriptor.WidthInBytes = copyWidth * gpuMat.elemSize();
  copyDescriptor.Height = copyHeight;
  if (stream) {
    CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
  } else {
    CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
  }
}

void copySurfaceToGpuMat(RHISurface::ptr surface, cv::cuda::GpuMat& gpuMat, const cv::cuda::Stream& stream) {
  copySurfaceToGpuMat(surface, gpuMat, (CUstream) stream.cudaPtr());
}

void copySurfaceToGpuMat(RHISurface::ptr surface, cv::cuda::GpuMat& gpuMat, CUstream stream) {
  assert(surface->isInteropSurface());

  size_t copyWidth = std::min<size_t>(surface->width(), gpuMat.cols);
  size_t copyHeight = std::min<size_t>(surface->height(), gpuMat.rows);

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(CUDA_MEMCPY2D));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.srcArray = (CUarray) surface->cudaArray();

  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_DEVICE;
  copyDescriptor.dstDevice = (CUdeviceptr) gpuMat.cudaPtr();
  copyDescriptor.dstPitch = gpuMat.step;

  copyDescriptor.WidthInBytes = copyWidth * gpuMat.elemSize();
  copyDescriptor.Height = copyHeight;
  if (stream) {
    CUDA_CHECK(cuMemcpy2DAsync(&copyDescriptor, stream));
  } else {
    CUDA_CHECK(cuMemcpy2D(&copyDescriptor));
  }
}


}; // namespace RHICUDA
#endif // HAVE_CUDA
