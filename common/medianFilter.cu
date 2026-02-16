#include "medianFilter.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/saturate_cast.hpp>
#include <opencv2/core/cuda/filters.hpp>

using namespace cv;
using namespace cv::cuda;

template <typename T> __global__ void medianFilter3x3_kernel(PtrStepSz<T> src, PtrStep<T> dst) {

  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= src.cols || y >= src.rows)
    return;

  // Offset the coordinates so that we'll always have a valid 3x3 region to walk over.
  // (This doesn't strictly match the median filter definition at the border, but it's close enough)
  int16_t xMin = std::min<int16_t>(std::max<int16_t>(x - 1, 0), src.cols - 3);
  int16_t yMin = std::min<int16_t>(std::max<int16_t>(y - 1, 0), src.rows - 3);

  T v[9];

  // Load all 9 samples from the source 3x3 neighborhood
  {
    T* samplePtr = v;
    for (int16_t kernelY = 0; kernelY < 3; ++kernelY) {
      const T* rowPtr = src.ptr(yMin + kernelY) + xMin;
      for (int16_t kernelX = 0; kernelX < 3; ++kernelX) {
        *(samplePtr++) =  rowPtr[kernelX];
      }
    }
  }

  // Reducing algorithm from <https://www.casual-effects.com/research/McGuire2008Median/median.pix>

  T temp;
  #define s2(a, b)                temp = a; a = std::min<T>(a, b); b = std::max<T>(temp, b);
  #define mn3(a, b, c)            s2(a, b); s2(a, c);
  #define mx3(a, b, c)            s2(b, c); s2(a, c);

  #define mnmx3(a, b, c)          mx3(a, b, c); s2(a, b);                                   // 3 exchanges
  #define mnmx4(a, b, c, d)        s2(a, b); s2(c, d); s2(a, c); s2(b, d);                  // 4 exchanges
  #define mnmx5(a, b, c, d, e)    s2(a, b); s2(c, d); mn3(a, c, e); mx3(b, d, e);           // 6 exchanges
  #define mnmx6(a, b, c, d, e, f) s2(a, d); s2(b, e); s2(c, f); mn3(a, b, c); mx3(d, e, f); // 7 exchanges

  // Starting with a subset of size 6, remove the min and max each time
  mnmx6(v[0], v[1], v[2], v[3], v[4], v[5]);
  mnmx5(v[1], v[2], v[3], v[4], v[6]);
  mnmx4(v[2], v[3], v[4], v[7]);
  mnmx3(v[3], v[4], v[8]);
  // Output is in v[4]
  dst.ptr(y)[x] = v[4];
}

void medianFilter3x3_u16(cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, CUstream stream) {
  dim3 block(8, 4);
  dim3 grid(
    cv::cuda::device::divUp(src.cols, block.x),
    cv::cuda::device::divUp(src.rows, block.y));

  medianFilter3x3_kernel<uint16_t><<<grid, block, 0, stream>>>(
    PtrStepSz<uint16_t>(src.rows, src.cols, (uint16_t*) src.cudaPtr(), src.step),
    PtrStep<uint16_t>((uint16_t*) dst.cudaPtr(), dst.step));
}

