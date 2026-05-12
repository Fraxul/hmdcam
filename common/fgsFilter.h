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
#pragma once
#include <cuda.h>
#include <opencv2/core/cuda.hpp>

// Fast Global Smoother filter on GPU.
//
// Algorithmic reference: Min et al. 2014 ("Fast Global Image Smoothing
// Based on Weighted Least Squares"). The 2D regularized smoothing problem
// is approximated by alternating 1D tridiagonal solves along rows and
// columns, with lambda decaying by lambdaAttenuation between iterations.
// Each 1D system is solved with the Thomas algorithm.
//
// guideTex: 2D texture object over the guide image. The caller is
//   responsible for setting up the texture (typically: unnormalized
//   coordinates, point sampling, returning a float per fetch).
//   sigmaColor is in the same units as the values returned by guideTex.
//   For a uint8 source read with cudaReadModeNormalizedFloat (returns
//   [0, 1]), pass sigmaColor / 255.f to match the conventional
//   8-bit-difference sigma scale.
//
// src, dst: GpuMats of matching size. Must both be CV_32FC1 or both be
//   CV_32FC2. CV_32FC2 runs the fused two-channel filter, in which the
//   same Thomas-elimination factorization (which depends only on lambda
//   and the guide) is applied to both data lanes in one pass --
//   appreciably cheaper than two CV_32FC1 calls. May alias.
//
// state: persistent scratch; the first call (or any time dimensions
//   change) does the allocation, subsequent calls are no-ops.
//
//
// Implementation strategy: hand-rolled partition method (Wang's algorithm /
// Spike-style) for the per-row tridiagonal solves. One warp per row;
// the 32 threads in the warp cooperate via warp shuffles to split the
// row into 32 segments, build local Mobius / affine maps, exchange
// boundary states with a log2(32) prefix scan, then apply per-step.
//
// This gives us high intra-row parallelism (32x more threads per
// row than thread-per-row Thomas) AND float2 fusion (shared interD
// factorization across both data lanes.
//
struct FGSFilterState;
void fgsFilter(
  FGSFilterState& state,
  CUtexObject guideTex,
  const cv::cuda::GpuMat& src,
  cv::cuda::GpuMat& dst,
  float lambda,
  float sigmaColor,
  float lambdaAttenuation,
  int numIterations,
  CUstream stream);

// Persistent scratch allocations for fgsFilter. The filter dimensions are
// expected to be stable between invocations, so this is constructed once
// (or each time dimensions change) and passed in on every call to keep
// allocations out of the hot path.
struct FGSFilterState {
  int width = 0;
  int height = 0;

  // H x W CV_32FC1, horizontal weights (image-native)
  cv::cuda::GpuMat Chor;

  // W x H CV_32FC1, vertical weights TRANSPOSED so the V
  // pass can be expressed as the H-pass kernel on a
  // transposed cur (no specialized V kernel needed).
  cv::cuda::GpuMat CvertT;

  // W x H CV_32FC1, transposed cur scratch (single-channel).
  cv::cuda::GpuMat workT;

  // W x H CV_32FC2, transposed cur scratch (two-channel
  // fused -- factorization is shared across both lanes).
  cv::cuda::GpuMat workT2;

  void ensureAllocated(int newWidth, int newHeight);
  void release();
};

// Pairwise out = num / (den + eps), single-channel. Same shape and
// stream as the existing fgsUnpackDivideEps, but operates on two
// separate CV_32FC1 inputs instead of an interleaved CV_32FC2.
void fgsDivideEpsPair(
  const cv::cuda::GpuMat& num,
  const cv::cuda::GpuMat& den,
  cv::cuda::GpuMat& out,
  float eps,
  CUstream stream);

// Pack (disp, conf) -> float2(disp*conf, conf) per pixel. Used to feed
// the two-channel fused FGS path for confidence-weighted disparity
// smoothing without an intermediate cv::cuda::multiply + merge pair.
void fgsPackDispConfMul(
  const cv::cuda::GpuMat& disp,
  const cv::cuda::GpuMat& conf,
  cv::cuda::GpuMat& out,
  CUstream stream);

// Inverse of fgsPackDispConfMul, applied to the filtered pair:
//   out = pair.x / (pair.y + eps)
// The eps floor guards against pixels that had near-zero filtered
// confidence (would otherwise blow up the divide).
void fgsUnpackDivideEps(
  const cv::cuda::GpuMat& pair,
  cv::cuda::GpuMat& out,
  float eps,
  CUstream stream);
