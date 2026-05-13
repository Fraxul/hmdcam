// This filter implementation was built by AI, referencing the algorithm implemented in the
// OpenCV ximgproc FastGlobalSmootherFilter (ximgproc/src/fgs_filter.cpp) as a baseline.
//
// While it is a complete rewrite, it seems only fair to assume that the OpenCV
// license of the referenced implementation would still apply; it is reproduced below.
//
// ---
//
// By downloading, copying, installing or using the software you agree to this license.
// If you do not agree to this license, do not download, install,
// copy or use the software.
//
//
// License Agreement
// For Open Source Computer Vision Library
// (3 - clause BSD License)
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met :
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and / or other materials provided with the distribution.
//
// * Neither the names of the copyright holders nor the names of the contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort(including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
#include "fgsFilter.h"
#include "rhi/cuda/CudaUtil.h"
#include <cuda_bf16.h>
#include <opencv2/core/cuda/common.hpp>
#include <algorithm>
#include <vector>

using cv::cuda::PtrStep;
using cv::cuda::PtrStepSz;
using cv::cuda::device::divUp;

namespace {

constexpr int kTransposeTile = 32;
constexpr float kEPS = 1e-43f;

// Component-wise helpers for the Thomas inner loop. Scalar (interD,
// coef) arithmetic is always float; the per-pixel data lane T is either
// float (single-channel) or float2 (two-channel fused filter, where the
// same factorization is reused for both lanes).
__device__ __forceinline__ float scaleInit(float cur, float invInit) {
  return cur * invInit;
}
__device__ __forceinline__ float2 scaleInit(float2 cur, float invInit) {
  return make_float2(cur.x * invInit, cur.y * invInit);
}

__device__ __forceinline__ float fwdStep(float cur, float curPrev, float coefPrev, float invDenom) {
  return (cur - curPrev * coefPrev) * invDenom;
}
__device__ __forceinline__ float2 fwdStep(float2 cur, float2 curPrev, float coefPrev, float invDenom) {
  return make_float2(
    (cur.x - curPrev.x * coefPrev) * invDenom,
    (cur.y - curPrev.y * coefPrev) * invDenom);
}

__device__ __forceinline__ float bwdStep(float cur, float interD, float curNext) {
  return cur - interD * curNext;
}
__device__ __forceinline__ float2 bwdStep(float2 cur, float interD, float2 curNext) {
  return make_float2(cur.x - interD * curNext.x, cur.y - interD * curNext.y);
}

// Tiled transpose with a +1 padded shared-memory tile to dodge bank
// conflicts on the read-back. Input is in.rows x in.cols; output is
// in.cols x in.rows.
template <typename T>
__global__ void transposeKernel(PtrStepSz<T> in, PtrStep<T> out) {
  __shared__ T tile[kTransposeTile][kTransposeTile + 1];

  const int x = blockIdx.x * kTransposeTile + threadIdx.x;
  const int y = blockIdx.y * kTransposeTile + threadIdx.y;
  if (x < in.cols && y < in.rows) {
    tile[threadIdx.y][threadIdx.x] = in.ptr(y)[x];
  }
  __syncthreads();

  const int outX = blockIdx.y * kTransposeTile + threadIdx.x; // column in output (= row in input)
  const int outY = blockIdx.x * kTransposeTile + threadIdx.y; // row in output    (= column in input)
  if (outX < in.rows && outY < in.cols) {
    out.ptr(outY)[outX] = tile[threadIdx.x][threadIdx.y];
  }
}

template <typename DispT>
__global__ void packDispConfMulKernel(PtrStepSz<DispT> disp, PtrStep<uint8_t> conf, PtrStep<float2> out, float scale) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (j >= disp.cols || i >= disp.rows) {
    return;
  }
  const float d = static_cast<float>(disp.ptr(i)[j]) * scale;
  const float c = static_cast<float>(conf.ptr(i)[j]) / 255.0f;
  out.ptr(i)[j] = make_float2(d * c, c);
}

template <typename T>
__global__ void unpackDivideScaleKernel(PtrStepSz<float2> pair, PtrStep<T> out, float scale) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (j >= pair.cols || i >= pair.rows) {
    return;
  }
  const float2 p = pair.ptr(i)[j];
  out.ptr(i)[j] = T((p.x / (p.y + kEPS)) * scale);
}

} // namespace

void fgsPackDispConfMul(
  const cv::cuda::GpuMat& disp,
  const cv::cuda::GpuMat& conf,
  float scale,
  cv::cuda::GpuMat& out,
  CUstream stream) {
  CV_Assert(disp.type() == CV_16UC1 && conf.type() == CV_8UC1);
  CV_Assert(disp.size() == conf.size());
  if (out.cols != disp.cols || out.rows != disp.rows || out.type() != CV_32FC2) {
    out.create(disp.rows, disp.cols, CV_32FC2);
  }
  dim3 block(32, 8);
  dim3 grid(divUp(disp.cols, static_cast<int>(block.x)), divUp(disp.rows, static_cast<int>(block.y)));
  packDispConfMulKernel<uint16_t><<<grid, block, 0, stream>>>(disp, conf, out, scale);
}

void fgsUnpackDivideScale(
  const cv::cuda::GpuMat& pair,
  cv::cuda::GpuMat& out,
  float scale,
  CUstream stream) {
  CV_Assert(pair.type() == CV_32FC2);
  if (out.cols != pair.cols || out.rows != pair.rows || out.type() != CV_16UC1) {
    out.create(pair.rows, pair.cols, CV_16UC1);
  }
  dim3 block(32, 8);
  dim3 grid(divUp(pair.cols, static_cast<int>(block.x)), divUp(pair.rows, static_cast<int>(block.y)));
  unpackDivideScaleKernel<uint16_t><<<grid, block, 0, stream>>>(pair, out, scale);
}

// ===== Partition-method FGS implementation =====

namespace {

// Build the horizontal-weight buffer in image-native row-major layout (no
// transpose). Used by the cuSPARSE path where weights feed a diagonal-
// builder kernel rather than an in-place Thomas pass.
__global__ void computeChorRowMajorKernel(CUtexObject guideTex, PtrStepSz<float> Chor, float sigmaColor) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x; // image column
  const int i = blockIdx.y * blockDim.y + threadIdx.y; // image row
  if (j >= Chor.cols || i >= Chor.rows) {
    return;
  }
  float w;
  if (j == Chor.cols - 1) {
    w = 0.0f;
  } else {
    const float p1 = tex2D<float>(guideTex, static_cast<float>(j) + 0.5f, static_cast<float>(i) + 0.5f);
    const float p2 = tex2D<float>(guideTex, static_cast<float>(j) + 1.5f, static_cast<float>(i) + 0.5f);
    w = -__expf(-fabsf(p1 - p2) / sigmaColor);
  }
  Chor.ptr(i)[j] = w;
}

// Build Cvert in TRANSPOSED layout: output is W rows of H cols, where
// CvertT[j][i] = vertical weight between guide(i, j) and guide(i+1, j).
// Storing transposed lets the V pass reuse the H-pass partition kernel
// after a cur transpose -- no specialized V kernel needed.
__global__ void computeCvertTKernel(CUtexObject guideTex, PtrStepSz<float> CvertT, float sigmaColor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x; // image row    (= col in transposed)
  const int j = blockIdx.y * blockDim.y + threadIdx.y; // image column (= row in transposed)
  const int imageHeight = CvertT.cols;
  const int imageWidth = CvertT.rows;
  if (i >= imageHeight || j >= imageWidth) {
    return;
  }
  float w;
  if (i == imageHeight - 1) {
    w = 0.0f;
  } else {
    const float p1 = tex2D<float>(guideTex, static_cast<float>(j) + 0.5f, static_cast<float>(i) + 0.5f);
    const float p2 = tex2D<float>(guideTex, static_cast<float>(j) + 0.5f, static_cast<float>(i) + 1.5f);
    w = -__expf(-fabsf(p1 - p2) / sigmaColor);
  }
  CvertT.ptr(j)[i] = w;
}

// Mobius map t -> (p*t + q) / (r*t + s), represented as the matrix
// [[p, q], [r, s]]. Composition (apply M_b after M_a) is the matrix
// product M_b * M_a.
//
// The transform is *projective*: scaling all four entries by a common
// non-zero scalar leaves the underlying map unchanged. We use this to
// renormalize after each composition; otherwise the entries grow as
// roughly k_j per step (~16001 at lambda=8000) and overflow fp32 after
// ~10 steps in a segment of 30.
__device__ __forceinline__ void mobiusNormalize(float& p, float& q, float& r, float& s) {
  const float m = fmaxf(fmaxf(fabsf(p), fabsf(q)), fmaxf(fabsf(r), fabsf(s)));
  const float inv = (m > 0.0f) ? (1.0f / m) : 1.0f;
  p *= inv;
  q *= inv;
  r *= inv;
  s *= inv;
}

// fp64 versions of the Mobius compose + scan. Per-step entries grow
// ~k_j (~16001 at lambda=8000), so a 30-step segment hits ~1e120 -- still
// inside fp64, but the warp scan then *multiplies* those segment
// matrices, and the cumulative entries reach 10^120*(tid+1). That blows
// past fp64's ~1.8e308 ceiling by tid=3. Renormalize on the projective
// scale after each compose to keep magnitudes bounded; fp64 still
// preserves precision in the smaller entries far better than fp32 did
// (the projective normalization itself isn't the precision-killer that
// it was in fp32 -- there the entries had only ~7 digits of mantissa
// to begin with).
__device__ __forceinline__ void mobiusNormalizeD(double& p, double& q, double& r, double& s) {
  const double m = fmax(fmax(fabs(p), fabs(q)), fmax(fabs(r), fabs(s)));
  const double inv = (m > 0.0) ? (1.0 / m) : 1.0;
  p *= inv;
  q *= inv;
  r *= inv;
  s *= inv;
}

__device__ __forceinline__ void mobiusComposeInplaceD(
  double& p, double& q, double& r, double& s,
  double pa, double qa, double ra, double sa) {
  double np = p * pa + q * ra;
  double nq = p * qa + q * sa;
  double nr = r * pa + s * ra;
  double ns = r * qa + s * sa;
  mobiusNormalizeD(np, nq, nr, ns);
  p = np;
  q = nq;
  r = nr;
  s = ns;
}

__device__ __forceinline__ void mobiusInclusiveScanWarpD(double& p, double& q, double& r, double& s) {
  const unsigned int mask = 0xffffffffu;
  for (int offset = 1; offset < 32; offset *= 2) {
    const double up_p = __shfl_up_sync(mask, p, offset);
    const double up_q = __shfl_up_sync(mask, q, offset);
    const double up_r = __shfl_up_sync(mask, r, offset);
    const double up_s = __shfl_up_sync(mask, s, offset);
    if (static_cast<int>(threadIdx.x) >= offset) {
      mobiusComposeInplaceD(p, q, r, s, up_p, up_q, up_r, up_s);
    }
  }
}

// Warp shuffle wrappers for the affine state's alpha component, which is
// T = float (single-channel) or T = float2 (fused two-channel). beta is
// always scalar float. For float2, shuffle the two lanes independently.
__device__ __forceinline__ float warpShflUp(unsigned mask, float v, int delta) {
  return __shfl_up_sync(mask, v, delta);
}
__device__ __forceinline__ float2 warpShflUp(unsigned mask, float2 v, int delta) {
  return make_float2(__shfl_up_sync(mask, v.x, delta), __shfl_up_sync(mask, v.y, delta));
}
__device__ __forceinline__ float warpShflDown(unsigned mask, float v, int delta) {
  return __shfl_down_sync(mask, v, delta);
}
__device__ __forceinline__ float2 warpShflDown(unsigned mask, float2 v, int delta) {
  return make_float2(__shfl_down_sync(mask, v.x, delta), __shfl_down_sync(mask, v.y, delta));
}

// Component-wise alpha_new = alpha_self + beta_self * alpha_other.
__device__ __forceinline__ float affineCombineAlpha(float self, float beta_self, float other) {
  return self + beta_self * other;
}
__device__ __forceinline__ float2 affineCombineAlpha(float2 self, float beta_self, float2 other) {
  return make_float2(self.x + beta_self * other.x, self.y + beta_self * other.y);
}

// Affine map x_new = step_alpha + step_beta * x_old. Composition (b after a):
//   (alpha_b + beta_b * alpha_a,  beta_b * beta_a)
// alpha is T (float or float2); beta is float (same factorization across lanes).
template <typename T>
__device__ __forceinline__ void affineComposeInplace(
  T& alpha, float& beta, // m_b (overwritten with m_b o m_a)
  T alpha_a, float beta_a) { // m_a (temporally earlier)
  alpha = affineCombineAlpha(alpha, beta, alpha_a);
  beta = beta * beta_a;
}

template <typename T>
__device__ __forceinline__ void affineInclusiveScanWarp(T& alpha, float& beta) {
  const unsigned int mask = 0xffffffffu;
  for (int offset = 1; offset < 32; offset *= 2) {
    const T up_alpha = warpShflUp(mask, alpha, offset);
    const float up_beta = __shfl_up_sync(mask, beta, offset);
    if (static_cast<int>(threadIdx.x) >= offset) {
      affineComposeInplace(alpha, beta, up_alpha, up_beta);
    }
  }
}

// Inclusive SUFFIX scan: thread t accumulates over threads t, t+1, ..., 31.
// Used by the backward substitution x[j] = y_j - interD[j] * x[j+1] which
// propagates right-to-left.
template <typename T>
__device__ __forceinline__ void affineInclusiveSuffixScanWarp(T& alpha, float& beta) {
  const unsigned int mask = 0xffffffffu;
  for (int offset = 1; offset < 32; offset *= 2) {
    const T dn_alpha = warpShflDown(mask, alpha, offset);
    const float dn_beta = __shfl_down_sync(mask, beta, offset);
    if (static_cast<int>(threadIdx.x) + offset < 32) {
      affineComposeInplace(alpha, beta, dn_alpha, dn_beta);
    }
  }
}

// One warp per row, 32 threads split the row into 32 segments. Each
// row is solved independently by its warp.
//
// Stage A (interD):
//   1) Each thread composes the Mobius matrix for its segment
//      (segment-local product of per-step matrices).
//   2) Warp scan -> exclusive prefix Mobius matrix at each thread.
//   3) Apply prefix to the row's boundary state (interD[-1] = 0) to
//      get the per-segment input boundary; then re-walk the segment
//      writing each interD[j] to memory.
//
// Stage B (forward x):
//   4) Each thread composes the affine map for its segment.
//   5) Warp scan -> exclusive prefix affine.
//   6) Apply -> per-segment x[t*S - 1] boundary; re-walk writing x_forward.
//
// Stage C (backward x):
//   7) Each thread composes the backward affine map for its segment.
//   8) Warp SUFFIX scan -> exclusive suffix affine.
//   9) Apply -> per-segment x[(t+1)*S] boundary; re-walk writing final x.
//
// All inter-thread communication is via warp shuffles -- no shared memory.
template <typename T>
__global__ void hPassPartitionKernel(
  PtrStepSz<T> cur, // H x W row-major
  PtrStepSz<float> Chor, // H x W row-major
  float lambda) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y; // image row
  if (i >= cur.rows) {
    return;
  }
  const int tid = threadIdx.x;
  const int W = cur.cols;
  const int S = (W + 31) / 32; // segment size (ceil division)
  const int segStart = tid * S;
  const int segEnd = min(segStart + S, W);

  // Pointer cache into row i of cur (final writeback target).
  T* const curRowGmem = cur.ptr(i);

  // SMEM cache layout per warp: [cur (W T)] [interD (W floats)] [Chor (W __nv_bfloat16s)].
  // Sorted in decreasing alignment requirement.
  // Chor is stored as bf16 -- 16-bit, but keeps fp32's 8-bit exponent so
  // values like exp(-30) (which arise for |p1-p2|/sigma >> 1 at edges)
  // stay representable instead of flushing to zero. fp16 was tried first
  // but flushed any weight below ~6e-8 to zero, which collapsed the
  // soft-edge behavior at high-contrast pixels. bf16's 7-bit mantissa
  // (~3 decimal digits) is plenty for these weights since the source has
  // only ~256 distinct |p1-p2| magnitudes.
  //
  // Halving Chor's footprint lets one more block fit per SM (SMEM is the
  // occupancy binder per ncu's advisory).
  //
  // Depending on W, we may need to pad at the end of the bfloat16 region.
  //
  extern __shared__ char smemRaw[];
  const int warpInBlock = threadIdx.y;

  // Round up warp stride to a multiple of 16 bytes -- realign after the bfloat16 region.
  const size_t warpStride = (((sizeof(__nv_bfloat16) + sizeof(float) + sizeof(T)) * W) + 15) & (~(15u));

  char* const warpSmem = smemRaw + warpInBlock * warpStride;
  T* const curRow = reinterpret_cast<T*>(warpSmem);
  float* const interDRow = reinterpret_cast<float*>(curRow + W);
  __nv_bfloat16* const smemChor = reinterpret_cast<__nv_bfloat16*>(interDRow + W);

  // Cooperative coalesced load of Chor + cur from GMEM into SMEM.
  for (int k = tid; k < W; k += 32) {
    smemChor[k] = __float2bfloat16(Chor.ptr(i)[k]);
    curRow[k] = curRowGmem[k];
  }
  __syncwarp();
  const __nv_bfloat16* const chorRow = smemChor;

  // --- Stage 1: local Mobius matrix for the segment.
  // Per-step matrix M_j = [[0, c_j], [-a_{j-1}, 1 - a_{j-1} - c_j]],
  // with a_{-1} = 0 and c_{W-1} = 0 from the boundary convention.
  //
  // Stage 1 in fp32 with per-step projective normalization. The compose
  // math here is well-conditioned even in fp32 -- the precision loss
  // only matters when these segment matrices are then multiplied
  // together in the scan, which we run in fp64.
  float coef_prev = (segStart > 0) ? lambda * __bfloat162float(chorRow[segStart - 1]) : 0.0f;
  float p = 1, q = 0, r = 0, s = 1;
#pragma unroll 4
  for (int j = segStart; j < segEnd; ++j) {
    const float c_j = lambda * __bfloat162float(chorRow[j]);
    const float k = 1.0f - coef_prev - c_j;
    float np = c_j * r;
    float nq = c_j * s;
    float nr = -coef_prev * p + k * r;
    float ns = -coef_prev * q + k * s;
    mobiusNormalize(np, nq, nr, ns);
    p = np;
    q = nq;
    r = nr;
    s = ns;
    coef_prev = c_j;
  }

  // --- Stage 2: cast to fp64, scan in fp64, then bring eq/es back to fp32.
  // The scan only adds log2(32)=5 composes, each preserving full fp64
  // precision -- avoids the cascade-of-small-entries problem that the
  // fp32 scan hit on pathological pixels.
  double pd = p, qd = q, rd = r, sd = s;
  mobiusInclusiveScanWarpD(pd, qd, rd, sd);
  double eq = __shfl_up_sync(0xffffffffu, qd, 1);
  double es = __shfl_up_sync(0xffffffffu, sd, 1);
  if (tid == 0) {
    eq = 0.0;
    es = 1.0;
  }

  // --- Stage 3: apply exclusive prefix to initial interD[-1] = 0,
  // then walk segment writing per-position interD values.
  // Mobius applied to 0: (er*0 + es)^-1 * (ep*0 + eq) = eq / es.
  float curInterD = static_cast<float>(eq / es);
  {
    float coefPrev3 = (segStart > 0) ? lambda * __bfloat162float(chorRow[segStart - 1]) : 0.0f;
#pragma unroll 4
    for (int j = segStart; j < segEnd; ++j) {
      const float c_j = lambda * __bfloat162float(chorRow[j]);
      const float denom = (1.0f - coefPrev3 - c_j) - curInterD * coefPrev3;
      curInterD = c_j / denom;
      interDRow[j] = curInterD;
      coefPrev3 = c_j;
    }
  }

  // Ensure all threads in the warp have published their interD writes
  // before any thread reads a neighbour's segment via interDRow[j-1].
  __syncwarp();

  // --- Stage 4: local affine map for forward x.
  // Per-step affine: x_j = (cur[j] - a_{j-1} * x_{j-1}) / d_j
  //                       = (cur[j] / d_j) + (-a_{j-1} / d_j) * x_{j-1}
  // alpha lives in T (matches the data lane); beta stays scalar -- the
  // factorization (d_j, a_{j-1}) is shared across both lanes for float2.
  T alpha = T();
  float beta = 1.0f;
  {
    float coefPrev4 = (segStart > 0) ? lambda * __bfloat162float(chorRow[segStart - 1]) : 0.0f;
    float interD_jm1 = (segStart > 0) ? interDRow[segStart - 1] : 0.0f;
#pragma unroll 4
    for (int j = segStart; j < segEnd; ++j) {
      const float c_j = lambda * __bfloat162float(chorRow[j]);
      const float d_j = (1.0f - coefPrev4 - c_j) - interD_jm1 * coefPrev4;
      const float invD = 1.0f / d_j;
      const T alpha_j = scaleInit(curRow[j], invD);
      const float beta_j = -coefPrev4 * invD;
      alpha = affineCombineAlpha(alpha_j, beta_j, alpha);
      beta = beta_j * beta;
      coefPrev4 = c_j;
      interD_jm1 = interDRow[j]; // = interD[j] (this iter's update)
    }
  }

  // --- Stage 5: inclusive scan, shift right for exclusive prefix.
  affineInclusiveScanWarp(alpha, beta);
  T eAlpha = warpShflUp(0xffffffffu, alpha, 1);
  float eBeta = __shfl_up_sync(0xffffffffu, beta, 1);
  if (tid == 0) {
    eAlpha = T();
    eBeta = 1; // identity: x[-1] = 0
  }
  (void) eBeta; // eBeta only used implicitly via curX = eAlpha (since x[-1] = 0).

  // --- Stage 6: apply, walking segment writing x_forward into cur.
  // x[segStart - 1] = eAlpha + eBeta * 0 = eAlpha.
  T curX = eAlpha;
  {
    float coefPrev6 = (segStart > 0) ? lambda * __bfloat162float(chorRow[segStart - 1]) : 0.0f;
    float interD_jm1 = (segStart > 0) ? interDRow[segStart - 1] : 0.0f;
#pragma unroll 4
    for (int j = segStart; j < segEnd; ++j) {
      const float c_j = lambda * __bfloat162float(chorRow[j]);
      const float d_j = (1.0f - coefPrev6 - c_j) - interD_jm1 * coefPrev6;
      const float invD = 1.0f / d_j;
      curX = fwdStep(curRow[j], curX, coefPrev6, invD);
      curRow[j] = curX;
      coefPrev6 = c_j;
      interD_jm1 = interDRow[j];
    }
  }

  __syncwarp();

  // --- Stage 7: local backward affine map (segment, right-to-left).
  // Step at j: x[j] = cur_forward[j] + (-interD[j]) * x[j+1]
  //          => alpha_step = cur_forward[j], beta_step = -interD[j].
  // Compose with running (alpha, beta) representing the map from
  // x[(t+1)*S] to the current step's output position.
  T bwAlpha = T();
  float bwBeta = 1.0f;
  for (int j = segEnd - 1; j >= segStart; --j) {
    const float interD_j = interDRow[j];
    const T y_j = curRow[j]; // forward value
    const float beta_step = -interD_j;
    // bwAlpha <- y_j + beta_step * bwAlpha
    bwAlpha = affineCombineAlpha(y_j, beta_step, bwAlpha);
    bwBeta = beta_step * bwBeta;
  }

  // --- Stage 8: inclusive suffix scan, shift down for exclusive suffix.
  affineInclusiveSuffixScanWarp(bwAlpha, bwBeta);
  T sufAlpha = warpShflDown(0xffffffffu, bwAlpha, 1);
  float sufBeta = __shfl_down_sync(0xffffffffu, bwBeta, 1);
  if (tid == 31) {
    sufAlpha = T();
    sufBeta = 1; // identity at rightmost segment
  }
  (void) sufBeta;

  // --- Stage 9: apply. Boundary state for thread t is x[(t+1)*S] = sufAlpha
  // + sufBeta * x[W]. For thread 31, x[W] is irrelevant: interD[W-1] = 0
  // makes beta_step = 0 at the rightmost step, which sets sufBeta to 0 on
  // every preceding thread, so the dependence on x[W] vanishes. Walking
  // right-to-left:
  T xNext = sufAlpha;
  for (int j = segEnd - 1; j >= segStart; --j) {
    const float interD_j = interDRow[j];
    const T y_j = curRow[j]; // forward value (SMEM)
    const T x_j = bwdStep(y_j, interD_j, xNext); // y_j - interD_j * xNext
    curRow[j] = x_j;
    xNext = x_j;
  }

  // Cooperative coalesced writeback to GMEM.
  __syncwarp();
  for (int k = tid; k < W; k += 32) {
    curRowGmem[k] = curRow[k];
  }
}

} // namespace

void FGSFilterState::ensureAllocated(int newWidth, int newHeight) {
  if (width == newWidth && height == newHeight) {
    return;
  }
  width = newWidth;
  height = newHeight;
  Chor.create(newHeight, newWidth, CV_32FC1);
  CvertT.create(newWidth, newHeight, CV_32FC1); // transposed: W rows of H cols
  workT.create(newWidth, newHeight, CV_32FC1);
  workT2.create(newWidth, newHeight, CV_32FC2); // float2 transposed-cur scratch
}

void FGSFilterState::release() {
  width = 0;
  height = 0;
  Chor.release();
  CvertT.release();
  workT2.release();
  workT.release();
}

void fgsFilter(
  FGSFilterState& state,
  CUtexObject guideTex,
  cv::cuda::GpuMat& src_dst,
  float lambda,
  float sigmaColor,
  float lambdaAttenuation,
  int numIterations,
  CUstream stream) {

  CV_Assert(!src_dst.empty());
  const int srcType = src_dst.type();
  CV_Assert(srcType == CV_32FC1 || srcType == CV_32FC2);

  const int W = src_dst.cols;
  const int H = src_dst.rows;
  state.ensureAllocated(W, H);

  // Build weight buffers (Chor row-major, CvertT transposed). One-shot per call;
  // both are reused across all numIterations passes.
  {
    dim3 block(32, 8);
    dim3 grid(divUp(W, static_cast<int>(block.x)), divUp(H, static_cast<int>(block.y)));
    computeChorRowMajorKernel<<<grid, block, 0, stream>>>(guideTex, state.Chor, sigmaColor);
  }
  {
    dim3 block(32, 8);
    dim3 grid(divUp(H, static_cast<int>(block.x)), divUp(W, static_cast<int>(block.y)));
    computeCvertTKernel<<<grid, block, 0, stream>>>(guideTex, state.CvertT, sigmaColor);
  }

  constexpr int kWarpsPerBlock = 4;
  const dim3 hBlock(32, kWarpsPerBlock), hGrid(1, divUp(H, kWarpsPerBlock));
  const dim3 vBlock(32, kWarpsPerBlock), vGrid(1, divUp(W, kWarpsPerBlock));
  const dim3 trBlock(kTransposeTile, kTransposeTile);
  const dim3 trGrid_fwd(divUp(W, kTransposeTile), divUp(H, kTransposeTile));
  const dim3 trGrid_back(divUp(H, kTransposeTile), divUp(W, kTransposeTile));

  // Dynamic SMEM size per block: per-warp (cur + interD + Chor). Chor is
  // stored in SMEM as bf16 (see kernel) to free up enough SMEM for one
  // more block per SM.
  //
  // We pad the warp SMEM size to a multiple of 16 bytes, which handles
  // alignment issues that could be caused by the bfloat16 section.
  CV_Assert((W & 1) == 0 && (H & 1) == 0);
  const size_t tBytes = (srcType == CV_32FC1) ? sizeof(float) : sizeof(float2);
  const size_t hSmemBytes = static_cast<size_t>(kWarpsPerBlock) * ((((sizeof(__nv_bfloat16) + sizeof(float) + tBytes) * W) + 15) & (~(15u)));
  const size_t vSmemBytes = static_cast<size_t>(kWarpsPerBlock) * ((((sizeof(__nv_bfloat16) + sizeof(float) + tBytes) * H) + 15) & (~(15u)));

  // The default per-kernel dynamic SMEM cap on Ampere is 48 KB; the
  // float2 H pass needs ~60 KB at W=960. Opt in to the architecture's
  // higher per-block limit, once per kernel instantiation. cudaFunc-
  // SetAttribute on the same kernel is a no-op after the first call.
  const int maxSmemBytes = static_cast<int>(std::max(hSmemBytes, vSmemBytes));
  if (srcType == CV_32FC1) {
    cudaFuncSetAttribute(&hPassPartitionKernel<float>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmemBytes);
  } else {
    cudaFuncSetAttribute(&hPassPartitionKernel<float2>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, maxSmemBytes);
  }

  // Per-iter dispatch: H pass on src_dst, transpose to workScratch, V pass on
  // workScratch with the transposed weights, transpose back. T (= float or
  // float2) picks the data lane; weights are always scalar.
  auto runIter = [&](auto tagT, cv::cuda::GpuMat& workScratch, float currentLambda) {
    using T = decltype(tagT);
    hPassPartitionKernel<T><<<hGrid, hBlock, hSmemBytes, stream>>>(src_dst, state.Chor, currentLambda);
    transposeKernel<T><<<trGrid_fwd, trBlock, 0, stream>>>(src_dst, workScratch);
    hPassPartitionKernel<T><<<vGrid, vBlock, vSmemBytes, stream>>>(workScratch, state.CvertT, currentLambda);
    transposeKernel<T><<<trGrid_back, trBlock, 0, stream>>>(workScratch, src_dst);
  };

  float currentLambda = lambda;
  for (int iter = 0; iter < numIterations; ++iter) {
    if (srcType == CV_32FC1) {
      runIter(float{}, state.workT, currentLambda);
    } else {
      runIter(float2{}, state.workT2, currentLambda);
    }
    currentLambda *= lambdaAttenuation;
  }
}
