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

template <unsigned int OversampleFactor>
__global__ void remapArrayRGB(CUtexObject lumaTex, CUtexObject chromaTex, CUtexObject undistortRectifyMap, PtrStepSz<uchar3> dst) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= dst.cols || y >= dst.rows)
    return;

  const float dx = 1.0f / static_cast<float>(OversampleFactor * dst.cols);
  const float dy = 1.0f / static_cast<float>(OversampleFactor * dst.rows);
  const float startX = static_cast<float>(x) / static_cast<float>(dst.cols);
  const float startY = static_cast<float>(y) / static_cast<float>(dst.rows);

  // Average Y and chroma across the oversample grid, in the source (YUV) domain.
  float accumY = 0.0f, accumCb = 0.0f, accumCr = 0.0f;
#pragma unroll
  for (uint yOffset = 0; yOffset < OversampleFactor; ++yOffset) {
#pragma unroll
    for (uint xOffset = 0; xOffset < OversampleFactor; ++xOffset) {
      float2 srcCoord = tex2D<float2>(undistortRectifyMap, startX + (dx * xOffset), startY + (dy * yOffset));
      accumY += tex2D<float>(lumaTex, srcCoord.x, srcCoord.y);
      float2 chroma = tex2D<float2>(chromaTex, srcCoord.x, srcCoord.y);
      accumCb += chroma.x;
      accumCr += chroma.y;
    }
  }

  const float invCount = 1.0f / static_cast<float>(OversampleFactor * OversampleFactor);
  const float yLuma = accumY * invCount;
  // Chroma is unsigned-normalized (0..1) with 0.5 == neutral; recenter to [-0.5, 0.5].
  const float cb = (accumCb * invCount) - 0.5f;
  const float cr = (accumCr * invCount) - 0.5f;

  // BT.601 full-range YUV->RGB.
  const float r = yLuma + (1.402f * cr);
  const float g = yLuma - (0.344136f * cb) - (0.714136f * cr);
  const float b = yLuma + (1.772f * cb);

  dst.ptr(y)[x] = make_uchar3(
    __float2uint_rn(255.0f * __saturatef(r)),
    __float2uint_rn(255.0f * __saturatef(g)),
    __float2uint_rn(255.0f * __saturatef(b)));
}

void remapArrayRGB(CUtexObject lumaTex, CUtexObject chromaTex, CUtexObject undistortRectifyMap, cv::cuda::GpuMat& dst, CUstream stream, unsigned int oversampleFactor) {
  assert(!dst.empty());
  assert(dst.type() == CV_8UC3);

  dim3 block(32, 8);
  dim3 grid(
    cv::cuda::device::divUp(dst.cols, block.x),
    cv::cuda::device::divUp(dst.rows, block.y));

  auto out = PtrStepSz<uchar3>(dst.rows, dst.cols, (uchar3*) dst.cudaPtr(), dst.step);

  switch (oversampleFactor) {
    case 1:
      remapArrayRGB<1><<<grid, block, 0, stream>>>(lumaTex, chromaTex, undistortRectifyMap, out);
      break;
    case 2:
      remapArrayRGB<2><<<grid, block, 0, stream>>>(lumaTex, chromaTex, undistortRectifyMap, out);
      break;
    case 3:
      remapArrayRGB<3><<<grid, block, 0, stream>>>(lumaTex, chromaTex, undistortRectifyMap, out);
      break;
    case 4:
      remapArrayRGB<4><<<grid, block, 0, stream>>>(lumaTex, chromaTex, undistortRectifyMap, out);
      break;
    default:
      assert(false && "Unhandled oversampleFactor");
  }
}
