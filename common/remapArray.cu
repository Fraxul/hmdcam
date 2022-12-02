#include "remapArray.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/saturate_cast.hpp>
#include <opencv2/core/cuda/filters.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
using namespace cv::cuda;

// borrowed from glm/gtc/packing.inl
static inline __host__  uint16_t packUnorm1x16(float s) {
  return static_cast<uint16_t>(roundf(std::min<float>(std::max<float>(s, 0.0f), 1.0f) * 65535.0f));
}
//static inline __device__ uint16_t packUnorm1x16(float s) {
//  return static_cast<uint16_t>(round(__saturatef(s) * 65535.0f));
//}

static inline __host__ __device__ float unpackUnorm1x16(uint16_t p) {
  float const Unpack(p);
  return Unpack * 1.5259021896696421759365224689097e-5f; // 1.0 / 65535.0
}

template <unsigned int DownsampleFactor> __global__ void remapArray(CUtexObject src, float2 srcDims, const PtrStep<ushort2> undistortRectifyMap, PtrStepSz<uchar> dst) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < dst.cols && y < dst.rows) {
    ushort2 unormCoords[DownsampleFactor * DownsampleFactor];
    #pragma unroll
    for (uint yOffset = 0; yOffset < DownsampleFactor; ++yOffset) {
      const ushort2* rowPtr = undistortRectifyMap.ptr((y * DownsampleFactor) + yOffset);
      #pragma unroll
      for (uint xOffset = 0; xOffset < DownsampleFactor; ++xOffset) {
        unormCoords[(yOffset * DownsampleFactor) + xOffset] = rowPtr[(x * DownsampleFactor) + xOffset];
      }
    }

    float samples[DownsampleFactor * DownsampleFactor];
    #pragma unroll
    for (uint i = 0; i < (DownsampleFactor * DownsampleFactor); ++i) {
      samples[i] = tex2D<float>(src,
        unpackUnorm1x16(unormCoords[i].x) * srcDims.x,
        unpackUnorm1x16(unormCoords[i].y) * srcDims.y);
    }

    float val = 0.0f;
    #pragma unroll
    for (uint i = 0; i < (DownsampleFactor * DownsampleFactor); ++i) {
      val += samples[i];
    }
    uchar b = __float2uint_rn(255.0f * __saturatef(val / static_cast<float>(DownsampleFactor * DownsampleFactor)));
    dst.ptr(y)[x] = b;
  }
}

void remapArray(CUtexObject src, cv::Size inputImageSize, cv::cuda::GpuMat& undistortRectifyMap, cv::cuda::GpuMat& dst, CUstream stream, unsigned int downsampleFactor) {
  assert(undistortRectifyMap.type() == CV_16UC2);
  dst.create(cv::Size(undistortRectifyMap.cols / downsampleFactor, undistortRectifyMap.rows / downsampleFactor), CV_8U);

  dim3 block(32, 8);
  dim3 grid(
    cv::cuda::device::divUp(dst.cols, block.x),
    cv::cuda::device::divUp(dst.rows, block.y));

  auto sz = make_float2(inputImageSize.width, inputImageSize.height);
  auto map = PtrStep<ushort2>((ushort2*) undistortRectifyMap.cudaPtr(), undistortRectifyMap.step);
  auto out = PtrStepSz<uchar>(dst.rows, dst.cols, (uchar*) dst.cudaPtr(), dst.step);

  switch (downsampleFactor) {
    case 1:
      remapArray<1><<<grid, block, 0, stream>>>(src, sz, map, out);
      break;
    case 2:
      remapArray<2><<<grid, block, 0, stream>>>(src, sz, map, out);
      break;
    case 3:
      remapArray<3><<<grid, block, 0, stream>>>(src, sz, map, out);
      break;
    case 4:
      remapArray<4><<<grid, block, 0, stream>>>(src, sz, map, out);
      break;
    default:
    assert(false && "Unhandled downsampleFactor");
  }
}

cv::cuda::GpuMat remapArray_initUndistortRectifyMap(cv::InputArray cameraMatrix, cv::InputArray distCoeffs, cv::InputArray rectification, cv::InputArray newProjection, cv::Size inputImageSize, unsigned int downsampleFactor) {

  cv::Mat xMat, yMat;
  cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, rectification, newProjection, inputImageSize, CV_32F, xMat, yMat);

  cv::Mat resMat;
  cv::Size tgtSize = cv::Size(inputImageSize.width / downsampleFactor, inputImageSize.height / downsampleFactor);
  resMat.create(tgtSize, CV_16UC2);

  float sampleScale = 1.0f / static_cast<float>(downsampleFactor * downsampleFactor);
  float texelBiasX = 0.5f;
  float texelBiasY = 0.5f;
  for (size_t y = 0; y < tgtSize.height; ++y) {
    for (size_t x = 0; x < tgtSize.width; ++x) {
      float fx = 0, fy = 0;
      for (size_t sampleY = 0; sampleY < downsampleFactor; ++sampleY) {
        for (size_t sampleX = 0; sampleX < downsampleFactor; ++sampleX) {
          fx += texelBiasX + xMat.ptr<float>((y * downsampleFactor) + sampleY)[(x * downsampleFactor) + sampleX];
          fy += texelBiasY + yMat.ptr<float>((y * downsampleFactor) + sampleY)[(x * downsampleFactor) + sampleX];
        }
      }

      // Convert to normalized texture coordinate range, pack into unorm16x2. packUnorm1x16 automatically clamps out-of-range values.
      resMat.ptr<ushort2>(y)[x] = make_ushort2(
        packUnorm1x16((fx * sampleScale) / static_cast<float>(inputImageSize.width)),
        packUnorm1x16((fy * sampleScale) / static_cast<float>(inputImageSize.height)));
    }
  }
  cv::cuda::GpuMat res;
  res.upload(resMat);
  return res;
}

