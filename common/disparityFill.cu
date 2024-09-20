#include "disparityFill.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>
#include <opencv2/core/cuda/saturate_cast.hpp>
#include <opencv2/core/cuda/filters.hpp>

using namespace cv;
using namespace cv::cuda;

struct alignas(uint32_t) DispChromaSample {
  union {
    struct {
      uint16_t disp;
      uint8_t chromaU;
      uint8_t chromaV;
    } data;

    uint32_t pack;
  };
};
static_assert(sizeof(DispChromaSample) == sizeof(uint32_t));

__device__ float chromaDistance2(DispChromaSample s1, DispChromaSample s2) {
  float du = static_cast<float>(s1.data.chromaU) - static_cast<float>(s2.data.chromaU);
  float dv = static_cast<float>(s1.data.chromaV) - static_cast<float>(s2.data.chromaV);
  float d2 = (du * du) + (dv * dv);
  return d2;
}

__device__ float chromaDistance2(DispChromaSample s1, uchar2 s2) {
  float du = static_cast<float>(s1.data.chromaU) - static_cast<float>(s2.x);
  float dv = static_cast<float>(s1.data.chromaV) - static_cast<float>(s2.y);
  float d2 = (du * du) + (dv * dv);
  return d2;
}

struct alignas(uint64_t) DispChromaMinMaxSample {
  DispChromaSample minSample;
  DispChromaSample maxSample;
};

static_assert(sizeof(DispChromaMinMaxSample) == sizeof(uint64_t));

inline __device__ void reduceSamples(DispChromaMinMaxSample& target, DispChromaMinMaxSample src, uint16_t maxValidDisparityRaw) {
  if (target.minSample.data.disp > maxValidDisparityRaw || target.minSample.data.disp > src.minSample.data.disp)
    target.minSample = src.minSample;

  if (target.maxSample.data.disp > maxValidDisparityRaw || target.maxSample.data.disp < src.maxSample.data.disp)
    target.maxSample = src.maxSample;
}

__global__ void __launch_bounds__(/*maxThreadsPerBlock=*/ 16) disparityFillDownsample1(CUtexObject chromaTex, uint16_t maxValidDisparityRaw,
  PtrStepSz<uint16_t> inDisparityMat,
  PtrStep<DispChromaMinMaxSample> outDisparityMinMaxMip1,
  PtrStep<DispChromaMinMaxSample> outDisparityMinMaxMip2,
  PtrStep<DispChromaMinMaxSample> outDisparityMinMaxMip3) {

  __shared__ DispChromaMinMaxSample wgSamples[4][4];

  const uint16_t mip1X = (blockDim.x * blockIdx.x + threadIdx.x);
  const uint16_t mip1Y = (blockDim.y * blockIdx.y + threadIdx.y);

  uint16_t dispSamples[4];

  // Clamp base coordinates to the edges of the input mat
  const uint16_t baseX = std::min<uint16_t>(mip1X * 2, inDisparityMat.cols - 1);
  const uint16_t baseY = std::min<uint16_t>(mip1Y * 2, inDisparityMat.rows - 1);

  dispSamples[0] = inDisparityMat.ptr(baseY + 0)[baseX + 0];
  dispSamples[1] = inDisparityMat.ptr(baseY + 0)[baseX + 1];
  dispSamples[2] = inDisparityMat.ptr(baseY + 1)[baseX + 0];
  dispSamples[3] = inDisparityMat.ptr(baseY + 1)[baseX + 1];

  // Operating under the assumption here that chroma is 2x the resolution of the disparity map, so there will be one chroma sample
  // for the first downsampling pass.
  // (disparity is 1/4 base resolution, chroma is 1/2 base resolution for NV12 format)
  // 0.5f offsets for texel centers.
  uchar2 chromaSampleBytes = tex2D<uchar2>(chromaTex, static_cast<float>(baseX) + 0.5f, static_cast<float>(baseY) + 0.5f);

  DispChromaMinMaxSample sample;

  // min-max
  sample.minSample.data.disp = 0xffffu;
  sample.minSample.data.chromaU = chromaSampleBytes.x;
  sample.minSample.data.chromaV = chromaSampleBytes.y;

  sample.maxSample.data.disp = 0;
  sample.maxSample.data.chromaU = chromaSampleBytes.x;
  sample.maxSample.data.chromaV = chromaSampleBytes.y;

  bool haveValidSample = false;
  for (int i = 0; i < 4; ++i) {
    if (dispSamples[i] <= maxValidDisparityRaw) {
      // valid sample, do min-max
      sample.minSample.data.disp = std::min<uint16_t>(sample.minSample.data.disp, dispSamples[i]);
      sample.maxSample.data.disp = std::max<uint16_t>(sample.maxSample.data.disp, dispSamples[i]);
      haveValidSample = true;
    }
  }

  if (!haveValidSample) {
    // all input disparity values were invalid -- pass through invalid disparity value to output.
    // Leave the chroma sample intact.
    sample.minSample.data.disp = 0xffffu;
    sample.maxSample.data.disp = 0xffffu;
  }

  // Write first mip
  wgSamples[threadIdx.y][threadIdx.x] = sample;
  __syncthreads(); // sync for workgroup write

  outDisparityMinMaxMip1.ptr(mip1Y)[mip1X] = sample;

  // Only use the upper 2x2 block of the threadgroup for the second mip
  if (threadIdx.x >= 2u || threadIdx.y >= 2u)
    return;

  // Each remaining thread gathers from a 2x2 region from the shared intermediary results
  int wgBaseX = threadIdx.x * 2;
  int wgBaseY = threadIdx.y * 2;

  sample =              wgSamples[wgBaseY + 0][wgBaseX + 0];
  reduceSamples(sample, wgSamples[wgBaseY + 0][wgBaseX + 1], maxValidDisparityRaw);
  reduceSamples(sample, wgSamples[wgBaseY + 1][wgBaseX + 0], maxValidDisparityRaw);
  reduceSamples(sample, wgSamples[wgBaseY + 1][wgBaseX + 1], maxValidDisparityRaw);
  __syncthreads(); // sync before workgroup write

  // Store to WG memory for final mip
  wgSamples[threadIdx.y][threadIdx.x] = sample;
  __syncthreads(); // sync for workgroup write

  // Store to mip2
  outDisparityMinMaxMip2.ptr((mip1Y / 2) + threadIdx.y)[(mip1X / 2) + threadIdx.x] = sample;

  // Only use the (0, 0) thread for the final mip
  if (threadIdx.x > 0 || threadIdx.y > 0)
    return;

  // Reduce and store final sample
  sample =              wgSamples[0][0];
  reduceSamples(sample, wgSamples[0][1], maxValidDisparityRaw);
  reduceSamples(sample, wgSamples[1][0], maxValidDisparityRaw);
  reduceSamples(sample, wgSamples[1][1], maxValidDisparityRaw);
  outDisparityMinMaxMip3.ptr(mip1Y / 4)[mip1X / 4] = sample;
}




__global__ void __launch_bounds__(/*maxThreadsPerBlock=*/ 16) disparityFillDownsample2(uint16_t maxValidDisparityRaw,
  PtrStepSz<DispChromaMinMaxSample> inDisparityMinMaxMat,
  PtrStep<DispChromaMinMaxSample> outDisparityMinMaxMip1,
  PtrStep<DispChromaMinMaxSample> outDisparityMinMaxMip2,
  PtrStep<DispChromaMinMaxSample> outDisparityMinMaxMip3) {

  __shared__ DispChromaMinMaxSample wgSamples[4][4];

  const uint16_t mip1X = (blockDim.x * blockIdx.x + threadIdx.x);
  const uint16_t mip1Y = (blockDim.y * blockIdx.y + threadIdx.y);

  // Clamp base coordinates to the edges of the input mat
  const uint16_t baseX = std::min<uint16_t>(mip1X * 2, inDisparityMinMaxMat.cols - 1);
  const uint16_t baseY = std::min<uint16_t>(mip1Y * 2, inDisparityMinMaxMat.rows - 1);

  DispChromaMinMaxSample baseSamples[4];
  baseSamples[0] = inDisparityMinMaxMat.ptr(baseY + 0)[baseX + 0];
  baseSamples[1] = inDisparityMinMaxMat.ptr(baseY + 0)[baseX + 1];
  baseSamples[2] = inDisparityMinMaxMat.ptr(baseY + 1)[baseX + 0];
  baseSamples[3] = inDisparityMinMaxMat.ptr(baseY + 1)[baseX + 1];

  DispChromaMinMaxSample sample = baseSamples[0];
  reduceSamples(sample, baseSamples[1], maxValidDisparityRaw);
  reduceSamples(sample, baseSamples[2], maxValidDisparityRaw);
  reduceSamples(sample, baseSamples[3], maxValidDisparityRaw);

  // Write first mip
  wgSamples[threadIdx.y][threadIdx.x] = sample;
  __syncthreads(); // sync for workgroup write

  outDisparityMinMaxMip1.ptr(mip1Y)[mip1X] = sample;

  // Only use the upper 2x2 block of the threadgroup for the second mip
  if (threadIdx.x >= 2u || threadIdx.y >= 2u)
    return;

  // Each remaining thread gathers from a 2x2 region from the shared intermediary results
  int wgBaseX = threadIdx.x * 2;
  int wgBaseY = threadIdx.y * 2;

  sample =              wgSamples[wgBaseY + 0][wgBaseX + 0];
  reduceSamples(sample, wgSamples[wgBaseY + 0][wgBaseX + 1], maxValidDisparityRaw);
  reduceSamples(sample, wgSamples[wgBaseY + 1][wgBaseX + 0], maxValidDisparityRaw);
  reduceSamples(sample, wgSamples[wgBaseY + 1][wgBaseX + 1], maxValidDisparityRaw);

  __syncthreads(); // sync before workgroup write

  // Store to WG memory for final mip
  wgSamples[threadIdx.y][threadIdx.x] = sample;
  __syncthreads(); // sync for workgroup write

  // Store to mip2
  outDisparityMinMaxMip2.ptr((mip1Y / 2) + threadIdx.y)[(mip1X / 2) + threadIdx.x] = sample;

  // Only use the (0, 0) thread for the final mip
  if (threadIdx.x > 0 || threadIdx.y > 0)
    return;

  // Reduce and store final sample
  sample =              wgSamples[0][0];
  reduceSamples(sample, wgSamples[0][1], maxValidDisparityRaw);
  reduceSamples(sample, wgSamples[1][0], maxValidDisparityRaw);
  reduceSamples(sample, wgSamples[1][1], maxValidDisparityRaw);
  outDisparityMinMaxMip3.ptr(mip1Y / 4)[mip1X / 4] = sample;
}


// Fill holes in tgtMat using samples from mipMat (which is one mip-level below tgtMat, or half of its dimensions)
__global__ void disparityFillUpsample(uint16_t maxValidDisparityRaw, PtrStepSz<DispChromaMinMaxSample> tgtMat, PtrStepSz<DispChromaMinMaxSample> mipMat) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= tgtMat.cols || y >= tgtMat.rows)
    return;

  DispChromaMinMaxSample sample = tgtMat.ptr(y)[x];
  if (sample.minSample.data.disp <= maxValidDisparityRaw)
    return; // Disparity is valid, nothing to do


  // Fetch the 3x3 sample neighborhood in the lower mip level
  int16_t xBase = (x / 2);
  int16_t yBase = (y / 2);

  // Clamp the lower mip level coordinates so that we'll always have a valid 3x3 region to walk over.
  // (the border texels will get extra distant samples, but that's probably OK)
  int16_t xMin = std::min<int16_t>(std::max<int16_t>(xBase - 1, 0), mipMat.cols - 3);
  int16_t yMin = std::min<int16_t>(std::max<int16_t>(yBase - 1, 0), mipMat.rows - 3);

  DispChromaMinMaxSample subSamples[9];
  for (int16_t kernelY = 0; kernelY < 3; ++kernelY) {
    for (int16_t kernelX = 0; kernelX < 3; ++kernelX) {
      subSamples[(kernelY * 3) + kernelX] = mipMat.ptr(yMin + kernelY)[xMin + kernelX];
    }
  }

  // Find shortest distance in chroma plane to populate min/max samples
  int shortestMinSampleIndex = -1;
  int shortestMaxSampleIndex = -1;
  float shortestMinSampleDistance = FLT_MAX;
  float shortestMaxSampleDistance = FLT_MAX;

  for (int i = 0; i < 9; ++i) {
    // Min sample population
    if (subSamples[i].minSample.data.disp <= maxValidDisparityRaw) {
      float d2 = chromaDistance2(sample.minSample, subSamples[i].minSample);
      if (d2 < shortestMinSampleDistance) {
        shortestMinSampleDistance = d2;
        shortestMinSampleIndex = i;
      }
    }

    // Max sample population
    if (subSamples[i].maxSample.data.disp <= maxValidDisparityRaw) {
      float d2 = chromaDistance2(sample.maxSample, subSamples[i].maxSample);
      if (d2 < shortestMaxSampleDistance) {
        shortestMaxSampleDistance = d2;
        shortestMaxSampleIndex = i;
      }
    }
  }

  if (shortestMinSampleIndex < 0 && shortestMaxSampleIndex < 0)
    return; // No replacement sample found

  sample.minSample = subSamples[shortestMinSampleIndex].minSample;
  sample.maxSample = subSamples[shortestMaxSampleIndex].maxSample;

  // Write the replacement disparity sample
  tgtMat.ptr(y)[x] = sample;
}

__global__ void disparityFillFinalMat(CUtexObject chromaTex, uint16_t maxValidDisparityRaw, PtrStepSz<uint16_t> disparityMat, PtrStepSz<DispChromaMinMaxSample> mipMat) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= disparityMat.cols || y >= disparityMat.rows)
    return;

  uint16_t centerDisp = disparityMat.ptr(y)[x];
  if (centerDisp < maxValidDisparityRaw)
    return; // Disparity is valid, nothing to do

  // Gather surrounding chroma samples. 0.5f offset for texel center in chroma target
  // divide-by-two is resolution scaler, assuming chroma resolution is 2x disparity resolution (half of base res, and disp is 1/4 base res)
  float cX = (x / 2) + 0.5f;
  float cY = (y / 2) + 0.5f;
  uchar2 centerChroma = tex2D<uchar2>(chromaTex, cX, cY);

  // Fetch the 3x3 sample neighborhood in the lower mip level
  int16_t xBase = (x / 2);
  int16_t yBase = (y / 2);

  // Clamp the lower mip level coordinates so that we'll always have a valid 3x3 region to walk over.
  // (the border texels will get extra distant samples, but that's probably OK)
  int16_t xMin = std::min<int16_t>(std::max<int16_t>(xBase - 1, 0), mipMat.cols - 3);
  int16_t yMin = std::min<int16_t>(std::max<int16_t>(yBase - 1, 0), mipMat.rows - 3);

  DispChromaMinMaxSample subSamples[9];
  for (int16_t kernelY = 0; kernelY < 3; ++kernelY) {
    for (int16_t kernelX = 0; kernelX < 3; ++kernelX) {
      subSamples[(kernelY * 3) + kernelX] = mipMat.ptr(yMin + kernelY)[xMin + kernelX];
    }
  }

  // Find the closest chroma match among all gathered min/max samples
  float chromaMinDist = FLT_MAX;
  for (int i = 0; i < 9; ++i) {
    if (subSamples[i].minSample.data.disp <= maxValidDisparityRaw) {
      float d2 = chromaDistance2(subSamples[i].minSample, centerChroma);
      if (d2 < chromaMinDist) {
        chromaMinDist = d2;
        centerDisp = subSamples[i].minSample.data.disp;
      }
    }
    if (subSamples[i].maxSample.data.disp <= maxValidDisparityRaw) {
      float d2 = chromaDistance2(subSamples[i].maxSample, centerChroma);
      if (d2 < chromaMinDist) {
        chromaMinDist = d2;
        centerDisp = subSamples[i].maxSample.data.disp;
      }
    }
  }

  // Write selected disparity
  disparityMat.ptr(y)[x] = centerDisp;
}

void disparityFill(CUtexObject chromaTex, cv::cuda::GpuMat& disparityMat, float maxValidDisparityRaw, std::vector<cv::cuda::GpuMat>& disparityMinMaxMips, CUstream stream) {
  // Downsample passes
  {
    dim3 block(4, 4);

    // First pass samples from the chroma surface and the raw input disparity map and produces 3 mips.
    // Block size is fixed at 4x4, grid size must be half of the input dimensions.
    // (each thread processes a 2x2 pixel block, and a thread block processes 8x8 pixels of the input)
    {
      dim3 grid(
        cv::cuda::device::divUp(disparityMat.cols / 2, block.x),
        cv::cuda::device::divUp(disparityMat.rows / 2, block.y));

      disparityFillDownsample1<<<grid, block, 0, stream>>>(chromaTex, static_cast<uint16_t>(maxValidDisparityRaw),
        PtrStepSz<uint16_t>(disparityMat.rows, disparityMat.cols, (uint16_t*) disparityMat.cudaPtr(), disparityMat.step),
        PtrStep<DispChromaMinMaxSample>((DispChromaMinMaxSample*) disparityMinMaxMips[0].cudaPtr(), disparityMinMaxMips[0].step),
        PtrStep<DispChromaMinMaxSample>((DispChromaMinMaxSample*) disparityMinMaxMips[1].cudaPtr(), disparityMinMaxMips[1].step),
        PtrStep<DispChromaMinMaxSample>((DispChromaMinMaxSample*) disparityMinMaxMips[2].cudaPtr(), disparityMinMaxMips[2].step));
    }

    // Subsequent passes operate solely on the MinMaxMip chain
    size_t passCount = disparityMinMaxMips.size() / 3;
    for (size_t passIdx = 1; passIdx < passCount; ++passIdx) {
      cv::cuda::GpuMat& inBase  = disparityMinMaxMips[(passIdx * 3) - 1];
      cv::cuda::GpuMat& outMip1 = disparityMinMaxMips[(passIdx * 3) + 0];
      cv::cuda::GpuMat& outMip2 = disparityMinMaxMips[(passIdx * 3) + 1];
      cv::cuda::GpuMat& outMip3 = disparityMinMaxMips[(passIdx * 3) + 2];

      dim3 grid(
        cv::cuda::device::divUp(inBase.cols / 2, block.x),
        cv::cuda::device::divUp(inBase.rows / 2, block.y));

      disparityFillDownsample2<<<grid, block, 0, stream>>>(static_cast<uint16_t>(maxValidDisparityRaw),
        PtrStepSz<DispChromaMinMaxSample>(inBase.rows, inBase.cols, (DispChromaMinMaxSample*) inBase.cudaPtr(), inBase.step),
        PtrStep<DispChromaMinMaxSample>((DispChromaMinMaxSample*) outMip1.cudaPtr(), outMip1.step),
        PtrStep<DispChromaMinMaxSample>((DispChromaMinMaxSample*) outMip2.cudaPtr(), outMip2.step),
        PtrStep<DispChromaMinMaxSample>((DispChromaMinMaxSample*) outMip3.cudaPtr(), outMip3.step));
    }
  }

  // Upsample passes: fill holes in larger mips from smaller mip levels
  {
    dim3 block(8, 8);
    ssize_t startMipIdx = static_cast<ssize_t>(disparityMinMaxMips.size()) - 2;

    for (ssize_t dstMipIdx = startMipIdx; dstMipIdx >= 0; --dstMipIdx) {
      cv::cuda::GpuMat& dstMip = disparityMinMaxMips[dstMipIdx];
      cv::cuda::GpuMat& srcMip = disparityMinMaxMips[dstMipIdx + 1];

      dim3 grid(
        cv::cuda::device::divUp(dstMip.cols, block.x),
        cv::cuda::device::divUp(dstMip.rows, block.y));

      disparityFillUpsample<<<grid, block, 0, stream>>>(static_cast<uint16_t>(maxValidDisparityRaw),
        PtrStepSz<DispChromaMinMaxSample>(dstMip.rows, dstMip.cols, (DispChromaMinMaxSample*) dstMip.cudaPtr(), dstMip.step),
        PtrStepSz<DispChromaMinMaxSample>(srcMip.rows, srcMip.cols, (DispChromaMinMaxSample*) srcMip.cudaPtr(), srcMip.step));
    }
  }
  // Final upsample pass: use the largest mip to fill holes in the disparity mat
  {
    dim3 block(32, 8);
    dim3 grid(
      cv::cuda::device::divUp(disparityMat.cols, block.x),
      cv::cuda::device::divUp(disparityMat.rows, block.y));

    cv::cuda::GpuMat& mip0 = disparityMinMaxMips[0];
    disparityFillFinalMat<<<grid, block, 0, stream>>>(chromaTex, static_cast<uint16_t>(maxValidDisparityRaw),
      PtrStepSz<uint16_t>(disparityMat.rows, disparityMat.cols, (uint16_t*) disparityMat.cudaPtr(), disparityMat.step),
      PtrStepSz<DispChromaMinMaxSample>(mip0.rows, mip0.cols, (DispChromaMinMaxSample*) mip0.cudaPtr(), mip0.step));
  }
}

