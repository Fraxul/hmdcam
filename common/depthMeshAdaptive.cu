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
__device__ inline float computeWeldedCornerDisparity(
  PtrStep<const uint16_t> disparity,
  PtrStep<const uint8_t> maxFlatLevel,
  int W, int H,
  int vx, int vy,
  int Lp) {
  int vx_c = min(vx, W - 1);
  int vy_c = min(vy, H - 1);
  float disp_raw = float(disparity.ptr(vy_c)[vx_c]);

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

  // V is on Q's boundary. Pick the appropriate edge and lerp between its endpoints.
  float disp_a, disp_b, t;
  if (vx == qx || vx == qx + szq) {
    int ex = min(vx, W - 1);
    int ey0 = qy;
    int ey1 = min(qy + szq, H - 1);
    disp_a = float(disparity.ptr(ey0)[ex]);
    disp_b = float(disparity.ptr(ey1)[ex]);
    t = float(vy - qy) / float(szq);
  } else {
    int ey = min(vy, H - 1);
    int ex0 = qx;
    int ex1 = min(qx + szq, W - 1);
    disp_a = float(disparity.ptr(ey)[ex0]);
    disp_b = float(disparity.ptr(ey)[ex1]);
    t = float(vx - qx) / float(szq);
  }
  return (1.0f - t) * disp_a + t * disp_b;
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
  PtrStepSz<const uint8_t> maxFlatLevel,
  PtrStep<const uint16_t> disparity,
  AdaptiveMeshVertex* outVerts,
  uint32_t* outIndices,
  DepthMeshAdaptiveCounters* counters,
  int W, int H,
  uint16_t discontinuityThresholdRaw,
  float cellOverlapMultiplier) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= W || y >= H) return;

  int enc = maxFlatLevel.ptr(y)[x];
  if (enc == 0) return;
  int L = enc - 1;
  int sz = 1 << L;

  // Anchor: top-left corner of the chosen block at level L
  if ((x & (sz - 1)) | (y & (sz - 1))) return;

  // The cell's representative disparity is the TL texel. It's also the welded value at
  // the TL corner by construction (the corner texel IS the cell's anchor texel), so the
  // TL corner is always welded; we only test the other three.
  float dRep = float(disparity.ptr(y)[x]);
  float threshold = float(discontinuityThresholdRaw);

  int xRight = x + sz;
  int yBottom = y + sz;
  float d00 = dRep;
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

  // Per-corner crack test: each non-TL corner's *welded* disparity is compared against
  // dRep. Testing the welded value (rather than the direct neighbor texel) catches
  // T-junction-lerp discontinuities the direct-texel test would miss -- e.g. when a
  // coarser neighbor Q matches dRep but Q's vertex-space edge lerp samples a texel
  // outside Q, producing a welded value mid-way between dRep and a totally unrelated
  // disparity. A missing/invalid neighbor (OOB, trim, or disparity over the valid
  // ceiling) also counts as a crack -- the welded value would be reading from
  // unfiltered raw disparity otherwise.
  bool d10Crack = !hasR || fabsf(d10 - dRep) > threshold;
  bool d01Crack = !hasB || fabsf(d01 - dRep) > threshold;
  bool d11Crack = !hasBR || fabsf(d11 - dRep) > threshold;

  // Propagate per-edge so corners on the same edge always crack together. Without this
  // step, e.g. d11 cracking alone (diagonal outlier) would leave the right and bottom
  // edges welded with one corner snapped to dRep -- the asymmetric tilt artifact.
  bool rightCrack = d10Crack || d11Crack;
  bool bottomCrack = d01Crack || d11Crack;
  if (rightCrack) d10 = dRep;
  if (bottomCrack) d01 = dRep;
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
  uint16_t dbg = uint16_t(L) | (rightCrack ? kAdaptiveDebugRightSnap : uint16_t(0)) | (bottomCrack ? kAdaptiveDebugBottomSnap : uint16_t(0));
  outVerts[vBase + 0] = {vxL, vyT, d00, dbg};
  outVerts[vBase + 1] = {vxR, vyT, d10, dbg};
  outVerts[vBase + 2] = {vxL, vyB, d01, dbg};
  outVerts[vBase + 3] = {vxR, vyB, d11, dbg};
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
      flatThresholdRaw);
  }

  // Pass 4: emit verts + indices for each anchor cell.
  {
    dim3 block(32, 4);
    dim3 grid(divUp(W, block.x), divUp(H, block.y));
    emitGeometryKernel<<<grid, block, 0, stream>>>(
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
