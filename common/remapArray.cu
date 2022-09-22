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

__global__ void remapArray(CUtexObject src, float2 srcDims, const PtrStep<ushort2> undistortRectifyMap, PtrStepSz<uchar> dst) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < dst.cols && y < dst.rows) {
    ushort2 unormCoords = undistortRectifyMap.ptr(y)[x];

    float rawSample = tex2D<float>(src,
        unpackUnorm1x16(unormCoords.x) * srcDims.x,
        unpackUnorm1x16(unormCoords.y) * srcDims.y);

    uchar b = __float2uint_rn(255.0f * __saturatef(rawSample));
    dst.ptr(y)[x] = b;
  }
}

template <typename DestT> __host__ void remapArray(CUtexObject src, float2 srcDims, const PtrStepSz<ushort2> undistortRectifyMap, void* dstCudaPtr, size_t dstStep, CUstream stream) {
  dim3 block(32, 8);
  dim3 grid(cv::cuda::device::divUp(undistortRectifyMap.cols, block.x), cv::cuda::device::divUp(undistortRectifyMap.rows, block.y));

  remapArray<<<grid, block, 0, stream>>>(src, srcDims, undistortRectifyMap, PtrStepSz<DestT>(undistortRectifyMap.rows, undistortRectifyMap.cols, (DestT*) dstCudaPtr, dstStep));
  cudaSafeCall( cudaGetLastError() );
}


void remapArray(CUtexObject src, cv::Size inputImageSize, cv::cuda::GpuMat undistortRectifyMap, cv::cuda::GpuMat dst, CUstream stream) {
  typedef void (*func_t)(CUtexObject, float2, PtrStepSz<ushort2>, void*, size_t, CUstream);
  static const func_t funcs[6][4] =
  {
      {remapArray<uchar>       , 0 /*remapArray<uchar2>*/ , 0 /*remapArray<uchar3>*/ , 0 /*remapArray<uchar4>*/ },
      {0 /*remapArray<schar>*/ , 0 /*remapArray<char2>*/  , 0 /*remapArray<char3>*/  , 0 /*remapArray<char4>*/  },
      {0 /*remapArray<ushort>*/, 0 /*remapArray<ushort2>*/, 0 /*remapArray<ushort3>*/, 0 /*remapArray<ushort4>*/},
      {0 /*remapArray<short>*/ , 0 /*remapArray<short2>*/ , 0 /*remapArray<short3>*/ , 0 /*remapArray<short4>*/ },
      {0 /*remapArray<int>*/   , 0 /*remapArray<int2>*/   , 0 /*remapArray<int3>*/   , 0 /*remapArray<int4>*/   },
      {0 /*remapArray<float>*/ , 0 /*remapArray<float2>*/ , 0 /*remapArray<float3>*/ , 0 /*remapArray<float4>*/ }
  };

  assert(undistortRectifyMap.type() == CV_16UC2);
  assert(dst.size() == undistortRectifyMap.size() && dst.cudaPtr() != nullptr);

  const func_t func = funcs[dst.depth()][dst.channels() - 1];
  if (!func)
      CV_Error(Error::StsUnsupportedFormat, "Unsupported output type");

  func(src, make_float2(inputImageSize.width, inputImageSize.height),
    PtrStepSz<ushort2>(undistortRectifyMap.rows, undistortRectifyMap.cols, (ushort2*) undistortRectifyMap.cudaPtr(), undistortRectifyMap.step),
    dst.cudaPtr(), dst.step, stream);
}

cv::cuda::GpuMat remapArray_initUndistortRectifyMap(cv::InputArray cameraMatrix, cv::InputArray distCoeffs, cv::InputArray rectification, cv::InputArray newProjection, cv::Size inputImageSize, unsigned int downsampleFactor) {

  cv::Mat xMat, yMat;
  cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, rectification, newProjection, inputImageSize, CV_32F, xMat, yMat);

  cv::Mat resMat;
  cv::Size tgtSize = cv::Size(inputImageSize.width / downsampleFactor, inputImageSize.height / downsampleFactor);
  resMat.create(tgtSize, CV_16UC2);

  float sampleScale = 1.0f / static_cast<float>(downsampleFactor * downsampleFactor);
  for (size_t y = 0; y < tgtSize.height; ++y) {
    for (size_t x = 0; x < tgtSize.width; ++x) {
      float fx = 0, fy = 0;
      for (size_t sampleY = 0; sampleY < downsampleFactor; ++sampleY) {
        for (size_t sampleX = 0; sampleX < downsampleFactor; ++sampleX) {
          fx += xMat.ptr<float>((y * downsampleFactor) + sampleY)[(x * downsampleFactor) + sampleX];
          fy += yMat.ptr<float>((y * downsampleFactor) + sampleY)[(x * downsampleFactor) + sampleX];
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

