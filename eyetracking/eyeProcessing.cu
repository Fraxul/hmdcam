#include "eyeProcessing.h"

#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda/common.hpp>
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/core/cuda/transform.hpp"
#include <cuda.h>
#include <cuda_fp16.h>

using namespace cv::cuda;
using namespace cv::cuda::device;

namespace cv { namespace cuda { namespace device {
  struct ApplyLUT8to16 : unary_function<uchar, ushort> {
    const ushort* lut;

    __host__ ApplyLUT8to16(const ushort* _lut) : lut(_lut) {}

    __device__ __forceinline__ ushort operator ()(uchar val) const {
        return lut[val];
    }
  };

  template <> struct TransformFunctorTraits<ApplyLUT8to16> : DefaultTransformFunctorTraits<ApplyLUT8to16> {
      enum { smart_shift = 4 };
  };
}}}

//void ApplyLUT8to16(const PtrStepSz<uchar>& src, const PtrStepSz<ushort>& dst, const ushort* lut, CUstream stream) {
//  device::transform(src, dst, cv::cuda::device::ApplyLUT8to16(lut), WithOutMask(), (cudaStream_t) stream);
//}

void ApplyLUT8to16(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, const ushort* lut, CUstream stream) {
  assert(src.type() == CV_8U);
  assert(dst.cols == src.cols && dst.rows == src.rows && dst.type() == CV_16U);

  device::transform(
    PtrStepSz<uchar>(src.rows, src.cols, (uchar*) src.cudaPtr(), src.step),
    PtrStepSz<ushort>(dst.rows, dst.cols, (ushort*) dst.cudaPtr(), dst.step),
    cv::cuda::device::ApplyLUT8to16(lut), WithOutMask(), (cudaStream_t) stream);
}

__global__ void deviceComputeClassIndex(const half* classWeights, uchar* outClassIndex, uint32_t width, uint32_t height) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (!(x < width && y < height))
    return;

  uint32_t planePitch = width * height;
  uint32_t offsetInPlane = (y * width) + x;

  // Best value and plane index
  half bestValue = classWeights[offsetInPlane];
  uchar bestPlaneIndex = 0;

  // Index of class with the highest value; first index wins.
  for (uchar planeIdx = 1; planeIdx < 4; ++planeIdx) {
    half planeCandidateValue = classWeights[(planeIdx * planePitch) + offsetInPlane];
    if (planeCandidateValue > bestValue) {
      bestValue = planeCandidateValue;
      bestPlaneIndex = planeIdx;
    }
  }

  outClassIndex[offsetInPlane] = bestPlaneIndex;
}

void ComputeClassIndex(const void* classWeightsFP16, uchar* outClassIndex, uint32_t width, uint32_t height, CUstream stream) {
  dim3 block(16, 8);
  dim3 grid(
    cv::cuda::device::divUp(width, block.x),
    cv::cuda::device::divUp(height, block.y));

  deviceComputeClassIndex<<<grid, block, 0, stream>>>(reinterpret_cast<const half*>(classWeightsFP16), outClassIndex, width, height);
}

