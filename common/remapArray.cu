#include "remapArray.h"
#include "opencv2/core/base.hpp"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/border_interpolate.hpp"
#include "opencv2/core/cuda/vec_traits.hpp"
#include "opencv2/core/cuda/vec_math.hpp"
#include "opencv2/core/cuda/saturate_cast.hpp"
#include "opencv2/core/cuda/filters.hpp"

using namespace cv;
using namespace cv::cuda;

__global__ void remapArray(CUtexObject src, const PtrStepf xMap, const PtrStepf yMap, PtrStepSz<uchar> dst) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < dst.cols && y < dst.rows) {
    float2 coords = make_float2(xMap.ptr(y)[x], yMap.ptr(y)[x]);
    dst.ptr(y)[x] = __float2int_rn(tex2D<float>(src, coords.x, coords.y) * 255.0f);
  }
}

template <typename DestT> __host__ void remapArray(CUtexObject src, const PtrStepSzf xMap, const PtrStepSzf yMap, void* dstCudaPtr, size_t dstStep, CUstream stream) {
  dim3 block(32, 8);
  dim3 grid(cv::cuda::device::divUp(xMap.cols, block.x), cv::cuda::device::divUp(xMap.rows, block.y));

  remapArray<<<grid, block, 0, stream>>>(src, xMap, yMap, PtrStepSz<DestT>(xMap.rows, xMap.cols, (DestT*) dstCudaPtr, dstStep));
  cudaSafeCall( cudaGetLastError() );
}


void remapArray(CUtexObject src, cv::cuda::GpuMat xMap, cv::cuda::GpuMat yMap, cv::cuda::GpuMat dst, CUstream stream) {
  typedef void (*func_t)(CUtexObject, PtrStepSzf, PtrStepSzf, void*, size_t, CUstream);
  static const func_t funcs[6][4] =
  {
      {remapArray<uchar>       , 0 /*remapArray<uchar2>*/ , 0 /*remapArray<uchar3>*/ , 0 /*remapArray<uchar4>*/ },
      {0 /*remapArray<schar>*/ , 0 /*remapArray<char2>*/  , 0 /*remapArray<char3>*/  , 0 /*remapArray<char4>*/  },
      {0 /*remapArray<ushort>*/, 0 /*remapArray<ushort2>*/, 0 /*remapArray<ushort3>*/, 0 /*remapArray<ushort4>*/},
      {0 /*remapArray<short>*/ , 0 /*remapArray<short2>*/ , 0 /*remapArray<short3>*/ , 0 /*remapArray<short4>*/ },
      {0 /*remapArray<int>*/   , 0 /*remapArray<int2>*/   , 0 /*remapArray<int3>*/   , 0 /*remapArray<int4>*/   },
      {0 /*remapArray<float>*/ , 0 /*remapArray<float2>*/ , 0 /*remapArray<float3>*/ , 0 /*remapArray<float4>*/ }
  };

  assert(xMap.type() == CV_32F && xMap.type() == CV_32F && xMap.size() == yMap.size());
  assert(dst.size() == xMap.size() && dst.cudaPtr() != nullptr);

  const func_t func = funcs[dst.depth()][dst.channels() - 1];
  if (!func)
      CV_Error(Error::StsUnsupportedFormat, "Unsupported output type");

  func(src,
    PtrStepSzf(xMap.rows, xMap.cols, (float*) xMap.cudaPtr(), xMap.step),
    PtrStepSzf(yMap.rows, yMap.cols, (float*) yMap.cudaPtr(), yMap.step),
    dst.cudaPtr(), dst.step, stream);
}


