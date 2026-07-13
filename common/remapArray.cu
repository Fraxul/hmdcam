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

// These kernels are texture-latency-bound: each output pixel does a dependent pair of texture
// fetches (rectification map -> source), and the source fetch cannot begin until the map fetch
// returns. Occupancy is already saturated, so the only lever left is instruction/memory-level
// parallelism within a thread: have each thread produce several independent output pixels so the
// compiler can keep many map-fetches (then many source-fetches) in flight at once, overlapping
// their latencies. TileX is that thread-coarsening factor -- the number of output columns a single
// thread emits, strided by blockDim.x so each warp still writes a coalesced run of columns.
//
// High oversample factors already issue OversampleFactor^2 independent fetches per pixel, so they
// hide latency on their own and gain little from extra tiling (which only adds register pressure /
// spills). We therefore scale TileX inversely with the oversample factor.
constexpr unsigned int kTileForOversample1 = 4;
constexpr unsigned int kTileForOversample2 = 1;
constexpr unsigned int kTileForOversample3 = 1;
constexpr unsigned int kTileForOversample4 = 1;

template <unsigned int OversampleFactor, unsigned int TileX>
__global__ void remapArray(CUtexObject src, CUtexObject undistortRectifyMap, PtrStepSz<uchar> dst) {
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (y >= dst.rows)
    return;

  // Base output column for this thread; the tile's remaining columns are strided by blockDim.x so a
  // warp covers blockDim.x consecutive columns per tile step (coalesced stores, local map fetches).
  const int baseX = (blockDim.x * TileX) * blockIdx.x + threadIdx.x;

  const float dx = 1.0f / static_cast<float>(OversampleFactor * dst.cols);
  const float dy = 1.0f / static_cast<float>(OversampleFactor * dst.rows);
  const float startY = static_cast<float>(y) / static_cast<float>(dst.rows);

  constexpr unsigned int kSamplesPerPixel = OversampleFactor * OversampleFactor;

  // Phase 1: gather every rectification-map coordinate for the whole tile. These fetches are
  // mutually independent, so they issue back-to-back and their latencies overlap.
  float2 coords[TileX][kSamplesPerPixel];
#pragma unroll
  for (unsigned int t = 0; t < TileX; ++t) {
    const int x = baseX + (t * blockDim.x);
    const float startX = static_cast<float>(x) / static_cast<float>(dst.cols);
#pragma unroll
    for (unsigned int yOffset = 0; yOffset < OversampleFactor; ++yOffset) {
#pragma unroll
      for (unsigned int xOffset = 0; xOffset < OversampleFactor; ++xOffset) {
        coords[t][(yOffset * OversampleFactor) + xOffset] =
          tex2D<float2>(undistortRectifyMap, startX + (dx * xOffset), startY + (dy * yOffset));
      }
    }
  }

  // Phase 2: sample the source at every gathered coordinate (again mutually independent), average
  // per pixel, and store. Bounds-check only the store so out-of-range tile columns cost nothing.
#pragma unroll
  for (unsigned int t = 0; t < TileX; ++t) {
    float val = 0.0f;
#pragma unroll
    for (unsigned int i = 0; i < kSamplesPerPixel; ++i)
      val += tex2D<float>(src, coords[t][i].x, coords[t][i].y);

    const int x = baseX + (t * blockDim.x);
    if (x < dst.cols)
      dst.ptr(y)[x] = __float2uint_rn(255.0f * __saturatef(val / static_cast<float>(kSamplesPerPixel)));
  }
}

void remapArray(CUtexObject src, CUtexObject undistortRectifyMap, cv::cuda::GpuMat& dst, CUstream stream, unsigned int oversampleFactor) {
  assert(!dst.empty());
  assert(dst.type() == CV_8U);

  dim3 block(32, 8);
  auto out = PtrStepSz<uchar>(dst.rows, dst.cols, (uchar*) dst.cudaPtr(), dst.step);

  const auto gridFor = [&](unsigned int tileX) {
    return dim3(
      cv::cuda::device::divUp(dst.cols, block.x * tileX),
      cv::cuda::device::divUp(dst.rows, block.y));
  };

  switch (oversampleFactor) {
    case 1:
      remapArray<1, kTileForOversample1><<<gridFor(kTileForOversample1), block, 0, stream>>>(src, undistortRectifyMap, out);
      break;
    case 2:
      remapArray<2, kTileForOversample2><<<gridFor(kTileForOversample2), block, 0, stream>>>(src, undistortRectifyMap, out);
      break;
    case 3:
      remapArray<3, kTileForOversample3><<<gridFor(kTileForOversample3), block, 0, stream>>>(src, undistortRectifyMap, out);
      break;
    case 4:
      remapArray<4, kTileForOversample4><<<gridFor(kTileForOversample4), block, 0, stream>>>(src, undistortRectifyMap, out);
      break;
    default:
      assert(false && "Unhandled oversampleFactor");
  }
}

template <unsigned int OversampleFactor, unsigned int TileX>
__global__ void remapArrayRGB(CUtexObject lumaTex, CUtexObject chromaTex, CUtexObject undistortRectifyMap, PtrStepSz<uchar3> dst) {
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  if (y >= dst.rows)
    return;

  const int baseX = (blockDim.x * TileX) * blockIdx.x + threadIdx.x;

  const float dx = 1.0f / static_cast<float>(OversampleFactor * dst.cols);
  const float dy = 1.0f / static_cast<float>(OversampleFactor * dst.rows);
  const float startY = static_cast<float>(y) / static_cast<float>(dst.rows);

  constexpr unsigned int kSamplesPerPixel = OversampleFactor * OversampleFactor;

  // Phase 1: gather all rectification-map coordinates for the tile (mutually independent fetches).
  float2 coords[TileX][kSamplesPerPixel];
#pragma unroll
  for (unsigned int t = 0; t < TileX; ++t) {
    const int x = baseX + (t * blockDim.x);
    const float startX = static_cast<float>(x) / static_cast<float>(dst.cols);
#pragma unroll
    for (unsigned int yOffset = 0; yOffset < OversampleFactor; ++yOffset) {
#pragma unroll
      for (unsigned int xOffset = 0; xOffset < OversampleFactor; ++xOffset) {
        coords[t][(yOffset * OversampleFactor) + xOffset] =
          tex2D<float2>(undistortRectifyMap, startX + (dx * xOffset), startY + (dy * yOffset));
      }
    }
  }

  const float invCount = 1.0f / static_cast<float>(kSamplesPerPixel);

  // Phase 2: sample luma + chroma at every coordinate, average in the YUV domain, convert, store.
#pragma unroll
  for (unsigned int t = 0; t < TileX; ++t) {
    float accumY = 0.0f, accumCb = 0.0f, accumCr = 0.0f;
#pragma unroll
    for (unsigned int i = 0; i < kSamplesPerPixel; ++i) {
      accumY += tex2D<float>(lumaTex, coords[t][i].x, coords[t][i].y);
      float2 chroma = tex2D<float2>(chromaTex, coords[t][i].x, coords[t][i].y);
      accumCb += chroma.x;
      accumCr += chroma.y;
    }

    const float yLuma = accumY * invCount;
    // Chroma is unsigned-normalized (0..1) with 0.5 == neutral; recenter to [-0.5, 0.5].
    const float cb = (accumCb * invCount) - 0.5f;
    const float cr = (accumCr * invCount) - 0.5f;

    // BT.601 full-range YUV->RGB.
    const float r = yLuma + (1.402f * cr);
    const float g = yLuma - (0.344136f * cb) - (0.714136f * cr);
    const float b = yLuma + (1.772f * cb);

    const int x = baseX + (t * blockDim.x);
    if (x < dst.cols)
      dst.ptr(y)[x] = make_uchar3(
        __float2uint_rn(255.0f * __saturatef(r)),
        __float2uint_rn(255.0f * __saturatef(g)),
        __float2uint_rn(255.0f * __saturatef(b)));
  }
}

void remapArrayRGB(CUtexObject lumaTex, CUtexObject chromaTex, CUtexObject undistortRectifyMap, cv::cuda::GpuMat& dst, CUstream stream, unsigned int oversampleFactor) {
  assert(!dst.empty());
  assert(dst.type() == CV_8UC3);

  dim3 block(32, 8);
  auto out = PtrStepSz<uchar3>(dst.rows, dst.cols, (uchar3*) dst.cudaPtr(), dst.step);

  const auto gridFor = [&](unsigned int tileX) {
    return dim3(
      cv::cuda::device::divUp(dst.cols, block.x * tileX),
      cv::cuda::device::divUp(dst.rows, block.y));
  };

  switch (oversampleFactor) {
    case 1:
      remapArrayRGB<1, kTileForOversample1><<<gridFor(kTileForOversample1), block, 0, stream>>>(lumaTex, chromaTex, undistortRectifyMap, out);
      break;
    case 2:
      remapArrayRGB<2, kTileForOversample2><<<gridFor(kTileForOversample2), block, 0, stream>>>(lumaTex, chromaTex, undistortRectifyMap, out);
      break;
    case 3:
      remapArrayRGB<3, kTileForOversample3><<<gridFor(kTileForOversample3), block, 0, stream>>>(lumaTex, chromaTex, undistortRectifyMap, out);
      break;
    case 4:
      remapArrayRGB<4, kTileForOversample4><<<gridFor(kTileForOversample4), block, 0, stream>>>(lumaTex, chromaTex, undistortRectifyMap, out);
      break;
    default:
      assert(false && "Unhandled oversampleFactor");
  }
}
