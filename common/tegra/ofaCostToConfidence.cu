#include "ofaCostToConfidence.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/saturate_cast.hpp>
#include <opencv2/core/cuda/filters.hpp>

using namespace cv;
using namespace cv::cuda;

__global__ void ofaCostToConfidenceKernel(CUtexObject ofaCost, int lowCostThreshold, int highCostThreshold, float costCurve, PtrStepSz<uint8_t> confidenceMat) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= confidenceMat.cols || y >= confidenceMat.rows)
    return;

  // Load from cost surface. 0.5f offsets for texel centers.
  int cost = tex2D<uint8_t>(ofaCost, static_cast<float>(x) + 0.5f, static_cast<float>(y) + 0.5f);

  // Linear remap between lowCostThreshold and highCostThreshold.
  // Semantically the thresholds are uint8_t (same as `cost`),
  // but we accept them as int so we can subtract without worrying about overflow.
  float t = static_cast<float>(cost - lowCostThreshold) /
    static_cast<float>(highCostThreshold - lowCostThreshold);

  // Confidence is the inverse of cost -- saturate, then flip the value.
  t = 1.0f - __saturatef(t);

  // Apply curve
  t = powf(t, costCurve);

  // Convert and write
  confidenceMat.ptr(y)[x] = __float2uint_rn(255.0f * t);
}

void ofaCostToConfidence(CUtexObject ofaCost, cv::cuda::GpuMat& confidenceMat, uint8_t lowCostThreshold, uint8_t highCostThreshold, float costCurve, CUstream stream) {
  dim3 block(32, 4);
  dim3 grid(
    cv::cuda::device::divUp(confidenceMat.cols, block.x),
    cv::cuda::device::divUp(confidenceMat.rows, block.y));

  ofaCostToConfidenceKernel<<<grid, block, 0, stream>>>(ofaCost, lowCostThreshold, highCostThreshold, costCurve,
    PtrStepSz<uint8_t>(confidenceMat.rows, confidenceMat.cols, (uint8_t*) confidenceMat.cudaPtr(), confidenceMat.step));
}
