#include "depthMeshAdaptive.h"
#include "rhi/cuda/CudaUtil.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <stdio.h>

using cv::cuda::PtrStep;
using cv::cuda::PtrStepSz;

namespace {

constexpr uint32_t kInvalidCell = 0xFFFFFFFFu;

__device__ __host__ inline uint32_t packCell(uint16_t mn, uint16_t mx) {
  return uint32_t(mn) | (uint32_t(mx) << 16);
}
__device__ __host__ inline uint16_t cellMin(uint32_t c) { return uint16_t(c & 0xFFFFu); }
__device__ __host__ inline uint16_t cellMax(uint32_t c) { return uint16_t(c >> 16); }
__device__ inline bool cellValid(uint32_t c) { return c != kInvalidCell; }

inline uint32_t divUp(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

// Packed anchor work-list entry: x in bits [0,12), y in bits [12,24), level in bits [24,28).
// 12 bits each comfortably covers the disparity grid (<4096 px per axis); 4 bits covers
// kAdaptiveMeshLevels. Keeps each work item a single 32-bit word for a coalesced read.
__device__ inline uint32_t packAnchor(int x, int y, int level) {
  return uint32_t(x) | (uint32_t(y) << 12) | (uint32_t(level) << 24);
}
__device__ inline void unpackAnchor(uint32_t a, int& x, int& y, int& level) {
  x = int(a & 0xFFFu);
  y = int((a >> 12) & 0xFFFu);
  level = int((a >> 24) & 0xFu);
}

// ----- Pyramid level 0: read disparity, mark trim region as invalid -----

__global__ void initLevel0Kernel(
  PtrStepSz<const uint16_t> disparity,
  PtrStep<uint32_t> outCells,
  uint16_t maxValidRaw,
  int trimLeft, int trimTop, int trimRightExclusive, int trimBottomExclusive) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= disparity.cols || y >= disparity.rows) return;

  bool inTrim = (x >= trimLeft) && (x < trimRightExclusive) && (y >= trimTop) && (y < trimBottomExclusive);
  uint16_t d = disparity.ptr(y)[x];
  bool ok = inTrim && (d <= maxValidRaw);

  outCells.ptr(y)[x] = ok ? packCell(d, d) : kInvalidCell;
}

// ----- 2x2 reduction: combine four child cells into one parent cell -----

__global__ void reduceLevelKernel(
  PtrStepSz<const uint32_t> in,
  PtrStepSz<uint32_t> out) {
  int ox = blockIdx.x * blockDim.x + threadIdx.x;
  int oy = blockIdx.y * blockDim.y + threadIdx.y;
  if (ox >= out.cols || oy >= out.rows) return;

  uint16_t mn = 0xFFFFu;
  uint16_t mx = 0u;
  bool allValid = true;

#pragma unroll
  for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
    for (int dx = 0; dx < 2; ++dx) {
      int sx = (ox << 1) + dx;
      int sy = (oy << 1) + dy;
      // Partial blocks at the right/bottom edge are treated as invalid so we never emit
      // a coarse quad that would extend past the disparity grid.
      if (sx >= in.cols || sy >= in.rows) {
        allValid = false;
        continue;
      }
      uint32_t c = in.ptr(sy)[sx];
      if (!cellValid(c)) {
        allValid = false;
        continue;
      }
      mn = min(mn, cellMin(c));
      mx = max(mx, cellMax(c));
    }
  }

  out.ptr(oy)[ox] = allValid ? packCell(mn, mx) : kInvalidCell;
}

// ----- Per-leaf-cell: pick the largest L such that the containing block is flat -----

struct PyramidLevels {
  PtrStep<uint32_t> level[kAdaptiveMeshLevels];
};

__global__ void computeMaxFlatLevelKernel(
  PyramidLevels py,
  PtrStepSz<uint8_t> outMaxFlatLevel,
  uint32_t* outWorkList,
  DepthMeshAdaptiveCounters* counters,
  uint16_t flatThreshold) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outMaxFlatLevel.cols || y >= outMaxFlatLevel.rows) return;

  int chosen = -1;
#pragma unroll
  for (int L = 0; L < kAdaptiveMeshLevels; ++L) {
    uint32_t c = py.level[L].ptr(y >> L)[x >> L];
    if (!cellValid(c)) break;
    // Subtraction in u16: if mx >= mn this is the range; if mx < mn (shouldn't happen
    // for valid cells) it underflows to a large value and fails the threshold.
    uint16_t range = uint16_t(cellMax(c) - cellMin(c));
    if (range > flatThreshold) break;
    chosen = L;
  }
  outMaxFlatLevel.ptr(y)[x] = uint8_t(chosen + 1); // 0 = skip

  // Compact emittable cells into the work list. A texel is the anchor for its chosen-level
  // block iff it's that block's top-left corner -- exactly the cells emitGeometryKernel emits
  // (it self-skips non-anchor texels via the same mask). Because flatness is monotonic across
  // levels, every texel in an L-flat region shares the same chosen level, so precisely one
  // (the top-left) appends here -- no double-counting. Emitting from this compacted list lets
  // the emit launch run one fully-active thread per cell instead of one per texel.
  if (chosen >= 0) {
    int sz = 1 << chosen;
    if (((x & (sz - 1)) | (y & (sz - 1))) == 0) {
      uint32_t slot = atomicAdd(&counters->anchorCount, 1u);
      outWorkList[slot] = packAnchor(x, y, chosen);
    }
  }
}

// ----- Per-corner welded disparity (T-junction snap) -----
//
// At each emitted corner V we look at the 4 quadrant cells around V. The cell inside
// our own patch self-skips because its level matches ours; among the remaining cells we
// pick the one whose containing patch is at the largest level coarser than ours (Q).
// If any such Q exists, V lies exactly on Q's boundary edge (proof: V's own cell is in
// our patch and the adjacent cell is in Q, so Q's extent must end precisely at V). We
// lerp the disparity along Q's shared edge between Q's two corner texture samples to
// produce a "welded" disparity at V that cells of either level can agree on. If no
// such Q exists, the welded value is just V's raw disparity texel.
//
// At Q's true corner the lerp evaluates to one of Q's endpoint samples (= V's sample),
// so the snap is a no-op. We rely on that rather than special-casing corners.
//
// The discontinuity threshold is applied separately in emitGeometryKernel, which
// compares this welded value to the cell's own representative disparity.
//
// De-recursed via a compile-time depth bound so the compiler can fully inline it (a
// self-recursive __device__ function is an inlining barrier -- nvcc emits a real call with a
// local-memory stack frame and heavy spills, which then shows up as long-scoreboard stalls).
// Each step welds to a STRICTLY coarser level (maxL > Lp) and levels cap at
// kAdaptiveMeshLevels-1, so the true recursion can descend at most kAdaptiveMeshLevels-1
// times. computeWeldedCornerDisparityD<Depth> calls <Depth-1>; the <0> base returns the raw
// sample. That base is only ever reached once the level has already saturated (where the
// original recursion also bottoms out and returns the raw sample), so the expression tree --
// and therefore the floating-point result -- is bit-identical to the recursive version.
template <int Depth>
__device__ inline float computeWeldedCornerDisparityD(
  PtrStep<const uint16_t> disparity,
  PtrStep<const uint8_t> maxFlatLevel,
  int W, int H,
  int vx, int vy,
  int Lp) {
  int vx_c = min(vx, W - 1);
  int vy_c = min(vy, H - 1);
  float disp_raw = float(disparity.ptr(vy_c)[vx_c]);

  if constexpr (Depth == 0) {
    return disp_raw; // depth cap: level has saturated, so this is always a leaf in valid data
  } else {
    int maxL = Lp;
    int qx = 0, qy = 0, szq = 0;

#pragma unroll
    for (int q = 0; q < 4; ++q) {
      int dx = (q & 1) - 1; // -1, 0, -1, 0
      int dy = ((q >> 1) & 1) - 1; // -1, -1, 0, 0
      int cx = vx + dx;
      int cy = vy + dy;
      if (cx < 0 || cy < 0 || cx >= W || cy >= H) continue;
      int Ln_enc = maxFlatLevel.ptr(cy)[cx];
      if (Ln_enc == 0) continue;
      int Ln = Ln_enc - 1;
      if (Ln > maxL) {
        maxL = Ln;
        int sz_n = 1 << Ln;
        qx = cx & ~(sz_n - 1);
        qy = cy & ~(sz_n - 1);
        szq = sz_n;
      }
    }

    if (maxL == Lp) return disp_raw; // no coarser neighbor -> raw sample

    // V is on Q's boundary. Lerp between Q's two corner values along the shared edge -- but the
    // corner values are computed RECURSIVELY at Q's level, not read raw. At a 3-level junction Q's
    // own corner is itself welded onto an even-coarser cell's edge, so Q renders that corner at its
    // welded value, not the raw texel; interpolating raw texels here would track a different line
    // than Q actually draws and crack the finer cell off Q's edge.
    float disp_a, disp_b, t;
    if (vx == qx || vx == qx + szq) {
      int ex = min(vx, W - 1);
      int ey0 = qy;
      int ey1 = min(qy + szq, H - 1);
      disp_a = computeWeldedCornerDisparityD<Depth - 1>(disparity, maxFlatLevel, W, H, ex, ey0, maxL);
      disp_b = computeWeldedCornerDisparityD<Depth - 1>(disparity, maxFlatLevel, W, H, ex, ey1, maxL);
      t = float(vy - qy) / float(szq);
    } else {
      int ey = min(vy, H - 1);
      int ex0 = qx;
      int ex1 = min(qx + szq, W - 1);
      disp_a = computeWeldedCornerDisparityD<Depth - 1>(disparity, maxFlatLevel, W, H, ex0, ey, maxL);
      disp_b = computeWeldedCornerDisparityD<Depth - 1>(disparity, maxFlatLevel, W, H, ex1, ey, maxL);
      t = float(vx - qx) / float(szq);
    }
    return (1.0f - t) * disp_a + t * disp_b;
  }
}

// Top-level entry: the worst-case descent from a level-0 cell is kAdaptiveMeshLevels-1 steps.
__device__ inline float computeWeldedCornerDisparity(
  PtrStep<const uint16_t> disparity,
  PtrStep<const uint8_t> maxFlatLevel,
  int W, int H,
  int vx, int vy,
  int Lp) {
  return computeWeldedCornerDisparityD<kAdaptiveMeshLevels - 1>(disparity, maxFlatLevel, W, H, vx, vy, Lp);
}

// Sample the raw disparity at the neighbor texel (nx, ny), returning false (and
// leaving outDisparity untouched) if the position is out of bounds or has no emitted
// cell at that texel. Used by the per-edge crack and overlap-extension tests.
__device__ inline bool sampleNeighborDisparity(
  PtrStep<const uint16_t> disparity,
  PtrStep<const uint8_t> maxFlatLevel,
  int W, int H,
  int nx, int ny,
  float& outDisparity) {
  if (nx < 0 || ny < 0 || nx >= W || ny >= H) return false;
  if (maxFlatLevel.ptr(ny)[nx] == 0) return false;
  outDisparity = float(disparity.ptr(ny)[nx]);
  return true;
}

// ----- Emit verts + indices for each anchor cell -----

__global__ void emitGeometryKernel(
  const uint32_t* workList,
  PtrStepSz<const uint8_t> maxFlatLevel,
  PtrStep<const uint16_t> disparity,
  AdaptiveMeshVertex* outVerts,
  uint32_t* outIndices,
  DepthMeshAdaptiveCounters* counters,
  int W, int H,
  uint16_t discontinuityThresholdRaw,
  float cellOverlapMultiplier) {
  uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
  // The launch is sized to the worst case (one anchor per texel); threads past the actual
  // anchor count form a contiguous tail and return immediately, so active warps stay full.
  if (item >= counters->anchorCount) return;

  // One thread per compacted anchor cell. The work list only holds true anchors, so the
  // per-texel validity (enc) and top-left-corner masks the old per-texel launch needed are
  // already guaranteed -- they're folded into computeMaxFlatLevelKernel's append.
  int x, y, L;
  unpackAnchor(workList[item], x, y, L);
  int sz = 1 << L;

  // dRep is this cell's representative disparity (its TL anchor texel); it's the value a
  // cracked (cliff-crossing) edge collapses to. All four corners -- TL included -- are
  // welded through the same pure position function, so any two cells meeting at a shared
  // corner compute the identical value and agree on it. (Previously TL was hardcoded to
  // dRep, which left a fine cell's TL un-welded wherever it lands mid-edge of a coarser
  // neighbor -- the source of the same-level T-junction cracks. In a flat region the TL
  // weld is a no-op: with no coarser neighbor it returns the anchor texel, i.e. dRep.)
  float dRep = float(disparity.ptr(y)[x]);
  float threshold = float(discontinuityThresholdRaw);

  int xRight = x + sz;
  int yBottom = y + sz;
  float d00 = computeWeldedCornerDisparity(disparity, maxFlatLevel, W, H, x, y, L);
  float d10 = computeWeldedCornerDisparity(disparity, maxFlatLevel, W, H, xRight, y, L);
  float d01 = computeWeldedCornerDisparity(disparity, maxFlatLevel, W, H, x, yBottom, L);
  float d11 = computeWeldedCornerDisparity(disparity, maxFlatLevel, W, H, xRight, yBottom, L);

  // Sample disparity at the texel just past each edge (and at the diagonal corner).
  // Direct samples drive extension; the corner-snap test below uses the welded values.
  float dN_L = 0.0f, dN_R = 0.0f, dN_T = 0.0f, dN_B = 0.0f, dN_BR = 0.0f;
  bool hasL = sampleNeighborDisparity(disparity, maxFlatLevel, W, H, x - 1, y, dN_L);
  bool hasR = sampleNeighborDisparity(disparity, maxFlatLevel, W, H, xRight, y, dN_R);
  bool hasT = sampleNeighborDisparity(disparity, maxFlatLevel, W, H, x, y - 1, dN_T);
  bool hasB = sampleNeighborDisparity(disparity, maxFlatLevel, W, H, x, yBottom, dN_B);
  bool hasBR = sampleNeighborDisparity(disparity, maxFlatLevel, W, H, xRight, yBottom, dN_BR);

  // Per-corner "bad" test: the welded value is unusable if it bridges a depth cliff (more
  // than threshold from dRep) or the neighbor cell anchoring that corner's texel is missing
  // (OOB/trim/over-ceiling), in which case the welded value is just unfiltered raw. d00 is
  // anchored by our own (valid) texel, so only the cliff test applies to it; the other three
  // keep the same missing-neighbor checks as before (R, B, and the BR diagonal).
  bool c00Bad = fabsf(d00 - dRep) > threshold;
  bool c10Bad = !hasR || fabsf(d10 - dRep) > threshold;
  bool c01Bad = !hasB || fabsf(d01 - dRep) > threshold;
  bool c11Bad = !hasBR || fabsf(d11 - dRep) > threshold;

  // An edge cracks if either endpoint is bad. Snapping *both* endpoints of every cracking
  // edge to dRep keeps that edge a flat segment at the cell's own depth (no quad stretched
  // across the cliff) and avoids the asymmetric-tilt artifact of snapping only one end. The
  // right/bottom decisions are byte-identical to the previous code; top/left are handled the
  // same way now that TL is welded and can itself bridge a cliff.
  bool topCrack = c00Bad || c10Bad;
  bool leftCrack = c00Bad || c01Bad;
  bool rightCrack = c10Bad || c11Bad;
  bool bottomCrack = c01Bad || c11Bad;
  if (topCrack || leftCrack) d00 = dRep;
  if (topCrack || rightCrack) d10 = dRep;
  if (leftCrack || bottomCrack) d01 = dRep;
  if (rightCrack || bottomCrack) d11 = dRep;

  // Per-edge overlap: extend C outward when the direct neighbor across an edge has
  // same-or-higher disparity (i.e. is closer to the camera) than dRep. Independent of
  // the corner-snap logic above -- extension is anchored to the direct neighbor texel,
  // not to T-junction lerp values, so the extrusion direction tracks "neighbor closer
  // => I extend past it". The extension is in q12.4 fixed-point so partial-cell
  // offsets (e.g. cellOverlapMultiplier = 1.1) survive serialization to uint16_t.
  int extQ = (cellOverlapMultiplier > 1.0f)
    ? int(roundf((cellOverlapMultiplier - 1.0f) * float(sz) * float(kAdaptiveMeshGridScale)))
    : 0;
  int extLQ = (extQ > 0 && hasL && (dN_L - dRep) >= threshold) ? extQ : 0;
  int extRQ = (extQ > 0 && hasR && (dN_R - dRep) >= threshold) ? extQ : 0;
  int extTQ = (extQ > 0 && hasT && (dN_T - dRep) >= threshold) ? extQ : 0;
  int extBQ = (extQ > 0 && hasB && (dN_B - dRep) >= threshold) ? extQ : 0;

  uint16_t vxL = uint16_t(max(0, x * kAdaptiveMeshGridScale - extLQ));
  uint16_t vyT = uint16_t(max(0, y * kAdaptiveMeshGridScale - extTQ));
  uint16_t vxR = uint16_t(xRight * kAdaptiveMeshGridScale + extRQ);
  uint16_t vyB = uint16_t(yBottom * kAdaptiveMeshGridScale + extBQ);

  uint32_t vBase = atomicAdd(&counters->vertexCounter, 4u);
  uint32_t iBase = atomicAdd(&counters->indexCounter, 6u);
  atomicAdd(&counters->levelHistograms[L], 1u);

#if ADAPTIVE_MESH_DEBUG
  uint16_t dbg = uint16_t(L) | (rightCrack ? kAdaptiveDebugRightSnap : uint16_t(0)) | (bottomCrack ? kAdaptiveDebugBottomSnap : uint16_t(0)) | (topCrack ? kAdaptiveDebugTopSnap : uint16_t(0)) | (leftCrack ? kAdaptiveDebugLeftSnap : uint16_t(0));
  // High bits carry the corner index (0=TL,1=TR,2=BL,3=BR) so the fragment shader can rebuild
  // a per-cell UV; the snap flags are identical across all four corners.
  outVerts[vBase + 0] = {vxL, vyT, d00, uint16_t(dbg | (uint16_t(0) << kAdaptiveDebugCornerShift))};
  outVerts[vBase + 1] = {vxR, vyT, d10, uint16_t(dbg | (uint16_t(1) << kAdaptiveDebugCornerShift))};
  outVerts[vBase + 2] = {vxL, vyB, d01, uint16_t(dbg | (uint16_t(2) << kAdaptiveDebugCornerShift))};
  outVerts[vBase + 3] = {vxR, vyB, d11, uint16_t(dbg | (uint16_t(3) << kAdaptiveDebugCornerShift))};
#else
  outVerts[vBase + 0] = {vxL, vyT, d00};
  outVerts[vBase + 1] = {vxR, vyT, d10};
  outVerts[vBase + 2] = {vxL, vyB, d01};
  outVerts[vBase + 3] = {vxR, vyB, d11};
#endif

  outIndices[iBase + 0] = vBase + 0;
  outIndices[iBase + 1] = vBase + 1;
  outIndices[iBase + 2] = vBase + 2;
  outIndices[iBase + 3] = vBase + 1;
  outIndices[iBase + 4] = vBase + 3;
  outIndices[iBase + 5] = vBase + 2;
}

// ----- Single-thread: fill in indirect-draw command counts -----

__global__ void writeIndirectArgsKernel(
  const DepthMeshAdaptiveCounters* counters,
  DrawElementsIndirectCommand* outArgs,
  DepthMeshAdaptiveCounters* outHostCounters) {
  if (blockIdx.x | threadIdx.x) return;
  uint32_t indexCount = counters->indexCounter;
  // Slot 0: stereo (2 instances). Slot 1: mono (1 instance).
  outArgs[0] = {indexCount, 2u, 0u, 0, 0u};
  outArgs[1] = {indexCount, 1u, 0u, 0, 0u};
  // Copy counters to host pinned memory.
  *outHostCounters = *counters;
}

} // namespace

// ----- DepthMeshAdaptiveScratch lifecycle -----

void DepthMeshAdaptiveScratch::allocate(uint32_t W, uint32_t H) {
  uint32_t lw = W;
  uint32_t lh = H;
  for (int L = 0; L < kAdaptiveMeshLevels; ++L) {
    mip[L].create(/*rows=*/ lh, /*cols=*/ lw, /*type=*/ CV_32S);
    lw = divUp(lw, 2);
    lh = divUp(lh, 2);
  }
  maxFlatLevel.create(/*rows=*/ H, /*cols=*/ W, /*type=*/ CV_8U);

  if (!d_workList)
    CUDA_CHECK(cuMemAlloc(&d_workList, size_t(W) * size_t(H) * sizeof(uint32_t)));
  if (!d_counters)
    CUDA_CHECK(cuMemAlloc(&d_counters, sizeof(DepthMeshAdaptiveCounters)));
  if (!h_counters) {
    CUDA_CHECK(cuMemHostAlloc((void**) &h_counters, sizeof(DepthMeshAdaptiveCounters), CU_MEMHOSTALLOC_DEVICEMAP));
    CUDA_CHECK(cuMemHostGetDevicePointer(&h_counters_devicePtr, h_counters, /*flags=*/ 0));
  }
}

void DepthMeshAdaptiveScratch::destroy() {
  for (int L = 0; L < kAdaptiveMeshLevels; ++L)
    mip[L].release();
  maxFlatLevel.release();
  CUDA_SAFE_FREE(d_workList);
  CUDA_SAFE_FREE(d_counters);
  CUDA_SAFE_FREE_HOST(h_counters);
  h_counters_devicePtr = 0;
}

// ----- Host entrypoint -----

void buildAdaptiveDepthMesh(
  const cv::cuda::GpuMat& disparityIn,
  uint16_t maxValidRaw,
  uint16_t flatThresholdRaw,
  uint16_t discontinuityThresholdRaw,
  float cellOverlapMultiplier,
  int trimLeft, int trimTop, int trimRight, int trimBottom,
  CUdeviceptr d_vbo,
  CUdeviceptr d_ibo,
  CUdeviceptr d_indirectArgs,
  DepthMeshAdaptiveScratch& scratch,
  CUstream stream) {
  assert(disparityIn.type() == CV_16U);
  const int W = disparityIn.cols;
  const int H = disparityIn.rows;

  // Reset device-side counters and histogram
  cuMemsetD8Async(scratch.d_counters, 0, sizeof(DepthMeshAdaptiveCounters), stream);

  // Pass 1: build pyramid level 0 (also applies the trim and validity test).
  {
    dim3 block(32, 4);
    dim3 grid(divUp(W, block.x), divUp(H, block.y));
    initLevel0Kernel<<<grid, block, 0, stream>>>(
      PtrStepSz<const uint16_t>(H, W, (const uint16_t*) disparityIn.cudaPtr(), disparityIn.step),
      PtrStep<uint32_t>((uint32_t*) scratch.mip[0].cudaPtr(), scratch.mip[0].step),
      maxValidRaw,
      trimLeft, trimTop,
      W - trimRight, H - trimBottom);
  }

  // Pass 2: reduce upward.
  for (int L = 1; L < kAdaptiveMeshLevels; ++L) {
    const auto& src = scratch.mip[L - 1];
    auto& dst = scratch.mip[L];
    dim3 block(32, 4);
    dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));
    reduceLevelKernel<<<grid, block, 0, stream>>>(
      PtrStepSz<const uint32_t>(src.rows, src.cols, (const uint32_t*) src.cudaPtr(), src.step),
      PtrStepSz<uint32_t>(dst.rows, dst.cols, (uint32_t*) dst.cudaPtr(), dst.step));
  }

  // Pass 3: per-leaf max flat level.
  {
    PyramidLevels py;
    for (int L = 0; L < kAdaptiveMeshLevels; ++L) {
      py.level[L] = PtrStep<uint32_t>((uint32_t*) scratch.mip[L].cudaPtr(), scratch.mip[L].step);
    }
    dim3 block(32, 4);
    dim3 grid(divUp(W, block.x), divUp(H, block.y));
    computeMaxFlatLevelKernel<<<grid, block, 0, stream>>>(
      py,
      PtrStepSz<uint8_t>(H, W, (uint8_t*) scratch.maxFlatLevel.cudaPtr(), scratch.maxFlatLevel.step),
      reinterpret_cast<uint32_t*>(scratch.d_workList),
      reinterpret_cast<DepthMeshAdaptiveCounters*>(scratch.d_counters),
      flatThresholdRaw);
  }

  // Pass 4: emit verts + indices for each anchor cell. Launched over the worst-case anchor
  // count (W*H); threads beyond the actual count (read from counters->anchorCount) early-out.
  {
    dim3 block(256);
    dim3 grid(divUp(uint32_t(W) * uint32_t(H), block.x));
    emitGeometryKernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(scratch.d_workList),
      PtrStepSz<const uint8_t>(H, W, (const uint8_t*) scratch.maxFlatLevel.cudaPtr(), scratch.maxFlatLevel.step),
      PtrStep<const uint16_t>((const uint16_t*) disparityIn.cudaPtr(), disparityIn.step),
      reinterpret_cast<AdaptiveMeshVertex*>(d_vbo),
      reinterpret_cast<uint32_t*>(d_ibo),
      reinterpret_cast<DepthMeshAdaptiveCounters*>(scratch.d_counters),
      W, H,
      discontinuityThresholdRaw,
      cellOverlapMultiplier);
  }

  // Pass 5: stamp the indirect draw commands with the resulting index count,
  // and write the host-side view of the counters (for debug stats).
  writeIndirectArgsKernel<<<1, 1, 0, stream>>>(
    reinterpret_cast<const DepthMeshAdaptiveCounters*>(scratch.d_counters),
    reinterpret_cast<DrawElementsIndirectCommand*>(d_indirectArgs),
    reinterpret_cast<DepthMeshAdaptiveCounters*>(scratch.h_counters_devicePtr));
}
