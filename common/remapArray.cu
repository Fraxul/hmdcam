#include "remapArray.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/saturate_cast.hpp>
#include <opencv2/core/cuda/filters.hpp>

using namespace cv;
using namespace cv::cuda;

template <unsigned int OversampleFactor>
__global__ void remapArray(CUtexObject src, CUtexObject undistortRectifyMap, PtrStepSz<uchar> dst) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  const float dx = 1.0f / static_cast<float>(OversampleFactor * dst.cols);
  const float dy = 1.0f / static_cast<float>(OversampleFactor * dst.rows);
  const float startX = static_cast<float>(x) / static_cast<float>(dst.cols);
  const float startY = static_cast<float>(y) / static_cast<float>(dst.rows);

  if (x < dst.cols && y < dst.rows) {
    float2 normalizedCoords[OversampleFactor * OversampleFactor];
#pragma unroll
    for (uint yOffset = 0; yOffset < OversampleFactor; ++yOffset) {
#pragma unroll
      for (uint xOffset = 0; xOffset < OversampleFactor; ++xOffset) {
        normalizedCoords[(yOffset * OversampleFactor) + xOffset] = tex2D<float2>(undistortRectifyMap, startX + (dx * xOffset), startY + (dy * yOffset));
      }
    }

    float samples[OversampleFactor * OversampleFactor];
#pragma unroll
    for (uint i = 0; i < (OversampleFactor * OversampleFactor); ++i) {
      samples[i] = tex2D<float>(src, normalizedCoords[i].x, normalizedCoords[i].y);
    }

    float val = 0.0f;
#pragma unroll
    for (uint i = 0; i < (OversampleFactor * OversampleFactor); ++i) {
      val += samples[i];
    }
    uchar b = __float2uint_rn(255.0f * __saturatef(val / static_cast<float>(OversampleFactor * OversampleFactor)));
    dst.ptr(y)[x] = b;
  }
}

void remapArray(CUtexObject src, CUtexObject undistortRectifyMap, cv::cuda::GpuMat& dst, CUstream stream, unsigned int oversampleFactor) {
  assert(!dst.empty());
  assert(dst.type() == CV_8U);

  dim3 block(32, 8);
  dim3 grid(
    cv::cuda::device::divUp(dst.cols, block.x),
    cv::cuda::device::divUp(dst.rows, block.y));

  auto out = PtrStepSz<uchar>(dst.rows, dst.cols, (uchar*) dst.cudaPtr(), dst.step);

  switch (oversampleFactor) {
    case 1:
      remapArray<1><<<grid, block, 0, stream>>>(src, undistortRectifyMap, out);
      break;
    case 2:
      remapArray<2><<<grid, block, 0, stream>>>(src, undistortRectifyMap, out);
      break;
    case 3:
      remapArray<3><<<grid, block, 0, stream>>>(src, undistortRectifyMap, out);
      break;
    case 4:
      remapArray<4><<<grid, block, 0, stream>>>(src, undistortRectifyMap, out);
      break;
    default:
      assert(false && "Unhandled oversampleFactor");
  }
}
