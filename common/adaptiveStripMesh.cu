#include "adaptiveStripMesh.h"
#include "rhi/cuda/CudaUtil.h"
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda/common.hpp>
#include <stdio.h>

using cv::cuda::PtrStep;
using cv::cuda::PtrStepSz;

namespace {

inline uint32_t divUp(uint32_t x, uint32_t y) { return (x + y - 1) / y; }

// ----- Pass 1: precompute the point-rendering quad corners -----

// Replicates TransformToLocalSpace() from MeshDisparityDepthMapCommon.h:
//   pp = (texcoord * inputImageSize + depthParameters.xy, depthParameters.z)
//   lw = depthParameters.w * (fDisp * mogrify.x)
//   local = R1 * (pp / lw)
// texcoord * inputImageSize collapses to gridPosition * mogrify (the disparity grid is the
// full image downsampled by mogrify), so everything stays in disparity-texel units here.
__device__ inline glm::vec3 stripLocalPosition(
  float gridX, float gridY, float invLw, const glm::vec4& depthParameters, const glm::mat3& R1, const glm::vec2& mogrify) {
  glm::vec3 pp(gridX * mogrify.x + depthParameters.x, gridY * mogrify.y + depthParameters.y, depthParameters.z);
  return R1 * (pp * invLw);
}

__device__ inline void writeStripMeshVertex(StripMeshVertex* outVertex, const glm::vec3& localPosition, float u, float v) {
  uint32_t packedTexCoords = __float2uint_rn(__saturatef(u) * 65535.0f) | (__float2uint_rn(__saturatef(v) * 65535.0f) << 16);
  *reinterpret_cast<float4*>(outVertex) = make_float4(localPosition.x, localPosition.y, localPosition.z, __uint_as_float(packedTexCoords));
}

// One thread per output vertex (4 threads share a cell), so each warp's float4 stores land
// contiguously -- the thread-per-cell version wrote at a 64-byte stride and was store-
// coalescing-bound. The four threads of a cell redundantly load the same disparity texel;
// that read broadcasts from L1.
// Corner order within a cell: [0] = left-top, [1] = left-bottom, [2] = right-top,
// [3] = right-bottom -- the same order as the point path's tristrip vertex stream, so the
// index pass can emit the identical winding.
__global__ void populateVertexBufferKernel(
  PtrStepSz<const uint16_t> disparity,
  float disparityPrescale,
  glm::vec4 depthParameters,
  glm::mat3 R1,
  glm::vec2 mogrify,
  float pointScale,
  glm::vec2 texCoordScale,
  int xOffset, int yOffset,
  StripMeshVertex* outVbo) {

  int vertexLinear = blockIdx.x * blockDim.x + threadIdx.x; // cell * 4 + corner, within the row
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = vertexLinear >> 2;
  int corner = vertexLinear & 3;
  if (x >= disparity.cols || y >= disparity.rows) return;

  // The fDisp floor matches the point shader: prevents divide-by-zero on zero-disparity
  // texels. Invalid cells still get well-defined (far away) positions; the strip pass never
  // emits indices referencing them.
  float fDisp = fmaxf(static_cast<float>(disparity.ptr(y)[x]) * disparityPrescale, 1.0f / 32.0f);
  float invLw = 1.0f / (depthParameters.w * (fDisp * mogrify.x));

  // Quad corners at [grid, grid + pointScale], matching the point shader's
  // quadCoordOffset * pointScale expansion. pointScale > 1 makes each strip overhang its
  // right/bottom neighbor, which covers the cracks left at strip breaks.
  float gridX = static_cast<float>(xOffset + x) + ((corner & 2) ? pointScale : 0.0f);
  float gridY = static_cast<float>(yOffset + y) + ((corner & 1) ? pointScale : 0.0f);

  writeStripMeshVertex(
    outVbo + (((y * disparity.cols) + x) * 4) + corner,
    stripLocalPosition(gridX, gridY, invLw, depthParameters, R1, mogrify),
    gridX * texCoordScale.x, gridY * texCoordScale.y);
}

// ----- Pass 2: walk row strips and emit merged-quad indices -----

struct RowWalkResult {
  uint32_t quadCount = 0;
  uint32_t stripCount = 0;
  uint32_t validCellCount = 0;
};

constexpr uint32_t kFullWarpMask = 0xFFFFFFFFu;
constexpr int kWalkWarpsPerBlock = 4;

// Rows per warp for the strip walk: each warp runs 32/kWalkGroupWidth row FSMs
// side-by-side in independent shuffle segments. Each warp instruction covers all its
// lanes, so replicating one row's FSM across 32 lanes wastes 31 of them; narrower groups
// retire proportionally more rows per instruction. Width 8 benched fastest on Orin at
// 480x270 (16 was within noise, 32 ~1.5x slower); width 4 can't exist -- the emit needs
// six lanes per group to distribute a quad's six index slots.
constexpr int kWalkGroupWidth = 8;
static_assert(kWalkGroupWidth >= 6 && (32 % kWalkGroupWidth) == 0);

// The merge test evaluates the angle between the current quad's direction
// (leading edge -> trailing edge) and the direction to the candidate trailing edge
// (leading edge -> next cell's trailing edge), both measured on the top corner verts.
// Because both vectors share the leading-edge origin, the deviation an accepted cell can
// introduce shrinks as the quad grows -- the test bounds how far the new endpoint strays
// from the quad's line, not just edge-to-edge curvature.
//
// The test runs in a 2D metric space instead of on the baked vertex positions. Within a
// row, every top-corner vertex is pos = s * (u, K, e) with s = 1/(dp.w * fDisp * mogrify.x)
// and u = gridX * mogrify.x + dp.x, where K = gridY * mogrify.y + dp.y and e = dp.z are
// row constants. Difference vectors therefore have the form (dp, dq*K, dq*e) with
// p = u * s, q = s, so dot(a, b) = dpA*dpB + m*dqA*dqB with metric m = K^2 + e^2 -- an
// exact isometry, and R1 (a pure rotation) never affects angles. This removes the float4
// vertex loads and all position shuffles from the walk: (p, q) derive from the disparity
// tile alone, and the pass no longer reads the VBO at all (pass 1 can overlap this pass).
//
// Greedy row walk, group-cooperative: kGroupWidth lanes run one row. The strip decision
// chain is inherently serial, so every lane of a group runs the same FSM with identical
// (replicated) state -- what the group parallelizes is data movement (each tile of
// kGroupWidth cells is loaded once, coalesced, and consumed through register shuffles)
// and the quad emit (lanes 0..5 each write one of the six indices). Group FSMs within a
// warp diverge in their decisions, so the FSM is branchless (predicated selects) to keep
// the warp converged for the width-segmented shuffles.
//
// Quads go directly into the caller's index buffer. Each row owns a deterministic,
// contiguous, x-ordered block [rowQuadBase, rowQuadBase + perRowStride): its used prefix
// holds the merged quads in x order and the unused tail is padded with degenerate quads
// (all six indices equal -> zero-area, culled). Welded neighbors land adjacent so the
// rasterizer's post-transform vertex-reuse window hits them; placement needs no atomic and
// no prefix sum -- rowQuadBase is a pure function of the row index.
// History: a thread-per-row serial walk was 1.5 ms on 480x270 (8 warps total,
// latency-bound); one warp per row with vec3 math from the VBO was ~430 us (issue-bound,
// ~120 instructions/cell); this shape is ~240 us solo / ~200 us overlapped with pass 1. An
// earlier variant packed quads through a shared atomic cursor to keep the buffer dense; the
// per-row layout draws a few degenerate pad quads but restored vertex reuse for a net ~8%
// faster render, and dropping the cursor bookkeeping sped the emit itself ~22%.
template <int kGroupWidth>
__device__ RowWalkResult walkRow(
  const uint16_t* disparityRow,
  int cellCount,
  uint32_t rowVertexBase,
  bool rowActive,
  float uRightBase, // (xOffset + pointScale) * mogrify.x + dp.x; lane cell's u = cell * mogrify.x + this
  float mogrifyX,
  float rowMetric, // m = K^2 + e^2
  float pointScaleTimesMogrifyX, // p_left = p_right - this * q
  float invLwDenominator, // dp.w * mogrify.x; s = 1 / (this * fDisp)
  float disparityPrescale,
  uint16_t maxValidRaw,
  uint16_t discontinuityThresholdRaw,
  float cosSquaredThreshold,
  uint32_t maxSkippedEdges,
  uint32_t rowQuadBase, // first quad slot owned by this row (rowIndex * perRowStride)
  uint32_t perRowStride, // quad slots reserved per row (== cellCount >= max quads/row)
  uint32_t* ibo) {

  const uint32_t subLane = threadIdx.x & static_cast<uint32_t>(kGroupWidth - 1);

  // Emit-lane constants: on emit, lane j of the group writes index slot j. The slot
  // pattern {lead, lead+1, trail, trail, lead+1, trail+1} (two triangles, same winding as
  // the point path's tristrip) is encoded as two bitmasks over j.
  const uint32_t laneUsesLead = (0x13u >> subLane) & 1u; // slots 0, 1, 4
  const uint32_t laneAddsOne = (0x32u >> subLane) & 1u; // slots 1, 4, 5

  RowWalkResult result;

  // quadSlot is the next ibo slot this group writes. It is group-uniform (all lanes start
  // from rowQuadBase and apply the same group-uniform increments), so the six emitting
  // lanes stay in lockstep without a shuffle.
  uint32_t quadSlot = rowQuadBase;

  int leadVertex = 0; // row-local index of the leading edge's top vert (its bottom pair is +1)
  float leadP = 0.0f, leadQ = 0.0f;
  int trailCell = -1; // < 0: no open quad. Otherwise the quad ends at this cell's right verts.
  float trailP = 0.0f, trailQ = 0.0f;
  uint32_t skippedEdges = 0;
  uint32_t previousDisparity = 0;

  for (int tileBase = 0; tileBase < cellCount; tileBase += kGroupWidth) {
    // Tile load: lane i owns cell tileBase + i and derives its (p_right, q) locally.
    // Keeping these as per-lane tile registers (instead of recomputing them from the
    // shuffled disparity) matters: shuffles of tile-resident values issue early, off the
    // FSM's serial chain, while a post-broadcast recompute would sit directly on it.
    const int laneCell = tileBase + static_cast<int>(subLane);
    uint32_t laneDisparity = 0xFFFFFFFFu; // out-of-range reads as invalid
    if (laneCell < cellCount)
      laneDisparity = disparityRow[laneCell];
    // fDisp floor matches pass 1; invalid disparities still produce finite garbage that
    // the FSM never consults.
    const float fDisp = fmaxf(static_cast<float>(laneDisparity) * disparityPrescale, 1.0f / 32.0f);
    const float laneQ = __fdividef(1.0f, invLwDenominator * fDisp);
    const float lanePRight = fmaf(static_cast<float>(laneCell), mogrifyX, uRightBase) * laneQ;

    // Serial scan over the tile. The shuffles are warp-collective (full mask, segmented
    // by kGroupWidth), so nothing between them may branch divergently -- all FSM updates
    // below are selects, and the emit is a predicated store.
    const int tileCells = min(kGroupWidth, cellCount - tileBase);
    for (int i = 0; i < tileCells; ++i) {
      const uint32_t d = __shfl_sync(kFullWarpMask, laneDisparity, i, kGroupWidth);
      const float candQ = __shfl_sync(kFullWarpMask, laneQ, i, kGroupWidth);
      const float candPRight = __shfl_sync(kFullWarpMask, lanePRight, i, kGroupWidth);
      const float candPLeft = fmaf(-pointScaleTimesMogrifyX, candQ, candPRight);
      const int x = tileBase + i;

      const bool valid = d <= maxValidRaw;
      const bool open = trailCell >= 0;

      // The open quad's trailing cell is always x - 1 (gaps close the strip above), so
      // previousDisparity is the trailing cell's disparity.
      const uint32_t step = (d > previousDisparity) ? (d - previousDisparity) : (previousDisparity - d);
      const bool crack = !open || (step > discontinuityThresholdRaw);

      // angle(v1, v2) < threshold, sqrt-free: dot > 0 restricts to acute angles, then
      // dot^2 >= cos^2(threshold) * |v1|^2 * |v2|^2. Operands are garbage when no quad is
      // open, but always finite, and the result is only consulted when open && !crack.
      const float dp1 = trailP - leadP, dq1 = trailQ - leadQ;
      const float dp2 = candPRight - leadP, dq2 = candQ - leadQ;
      const float mdq1 = rowMetric * dq1;
      const float d12 = fmaf(dp1, dp2, mdq1 * dq2);
      const float n1 = fmaf(dp1, dp1, mdq1 * dq1);
      const float n2 = fmaf(dp2, dp2, rowMetric * dq2 * dq2);
      const bool angleOK = (d12 > 0.0f) && ((d12 * d12) >= (cosSquaredThreshold * n1 * n2));
      const bool merge = valid && !crack && (skippedEdges < maxSkippedEdges) && angleOK;

      // Close the open quad on: invalid cell, discontinuity crack, or curvature break.
      const bool emitNow = open && !merge;
      const uint32_t trailVertex = static_cast<uint32_t>(trailCell * 4 + 2);
      const uint32_t emitIndex = rowVertexBase + (laneUsesLead ? static_cast<uint32_t>(leadVertex) : trailVertex) + laneAddsOne;

      // quadSlot walks straight through the row's own block, so the emit is a plain
      // predicated store with no cursor bookkeeping on the serial chain. quadSlot stays
      // group-uniform because doEmit is group-uniform (FSM state is replicated per group).
      const bool doEmit = emitNow && rowActive;
      if (doEmit && (subLane < 6u))
        ibo[(quadSlot * 6) + subLane] = emitIndex;
      quadSlot += doEmit ? 1u : 0u;
      result.quadCount += emitNow;

      // A valid cell always reopens: crack re-leads at this cell's left verts (intentional
      // gap, covered by the pointScale overlap, same as the point path); a curvature break
      // welds by re-leading at the old trailing verts; a merge keeps the lead.
      const bool isCrack = valid && crack;
      const bool isWeldBreak = valid && !crack && !merge;
      leadVertex = isCrack ? (x * 4) : (isWeldBreak ? static_cast<int>(trailVertex) : leadVertex);
      leadP = isCrack ? candPLeft : (isWeldBreak ? trailP : leadP);
      leadQ = isCrack ? candQ : (isWeldBreak ? trailQ : leadQ);
      skippedEdges = merge ? (skippedEdges + 1) : 0;
      result.stripCount += isCrack;
      result.validCellCount += valid;
      trailCell = valid ? x : -1;
      trailP = valid ? candPRight : trailP;
      trailQ = valid ? candQ : trailQ;
      previousDisparity = valid ? d : previousDisparity;
    }
  }
  if (trailCell >= 0) {
    const uint32_t trailVertex = static_cast<uint32_t>(trailCell * 4 + 2);
    const uint32_t emitIndex = rowVertexBase + (laneUsesLead ? static_cast<uint32_t>(leadVertex) : trailVertex) + laneAddsOne;
    if (rowActive) {
      if (subLane < 6u)
        ibo[(quadSlot * 6) + subLane] = emitIndex;
      ++quadSlot;
    }
    ++result.quadCount;
  }

  // Pad the unwritten tail of this row's reserved block with degenerate quads (all six
  // indices equal -> zero-area triangles, culled) so the whole drawn range is populated.
  // quadSlot is group-uniform; divergent across groups but no shuffle is involved.
  if (rowActive) {
    const uint32_t rowQuadEnd = rowQuadBase + perRowStride;
    while (quadSlot < rowQuadEnd) {
      if (subLane < 6u)
        ibo[(quadSlot * 6) + subLane] = 0;
      ++quadSlot;
    }
  }

  return result;
}

// One row per kGroupWidth-lane group, 32/kGroupWidth rows per warp. Each row emits straight
// into its own deterministic block of the caller's index buffer; the whole mesh renders as
// a single indirect draw whose (deterministic) index count the tail kernel stamps.
template <int kGroupWidth>
__global__ void emitStripsKernel(
  PtrStepSz<const uint16_t> disparity,
  uint32_t* outIndices,
  StripMeshCounters* counters,
  float disparityPrescale,
  glm::vec4 depthParameters,
  glm::vec2 mogrify,
  float pointScale,
  int xOffset, int yOffset,
  uint16_t maxValidRaw,
  uint16_t discontinuityThresholdRaw,
  float cosSquaredThreshold,
  uint32_t maxSkippedEdges) {

  constexpr int kGroupsPerWarp = 32 / kGroupWidth;
  const int warpIndex = (blockIdx.x * kWalkWarpsPerBlock) + static_cast<int>(threadIdx.x >> 5);
  const int group = static_cast<int>(threadIdx.x & 31u) / kGroupWidth;
  const int rowIndex = (warpIndex * kGroupsPerWarp) + group;

  // Tail groups redundantly re-walk the last row with all writes suppressed, instead of
  // exiting: the walk's segmented shuffles are warp-collective, so every group must run
  // the full loop.
  const bool rowActive = rowIndex < disparity.rows;
  const int y = min(rowIndex, disparity.rows - 1);

  const int cellCount = disparity.cols;
  const float K = fmaf(static_cast<float>(yOffset + y), mogrify.y, depthParameters.y);
  const float rowMetric = fmaf(K, K, depthParameters.z * depthParameters.z);
  const float uRightBase = fmaf(static_cast<float>(xOffset) + pointScale, mogrify.x, depthParameters.x);

  // Row-contiguous placement: each row owns [rowIndex * cellCount, +cellCount). A row emits
  // at most cellCount quads (>= 1 cell per quad), so cellCount is a safe stride. Uses the
  // real rowIndex (not the clamped y) so inactive tail groups never alias a valid row.
  const uint32_t rowQuadBase = static_cast<uint32_t>(rowIndex) * static_cast<uint32_t>(cellCount);
  const uint32_t perRowStride = static_cast<uint32_t>(cellCount);

  RowWalkResult walked = walkRow<kGroupWidth>(
    disparity.ptr(y), cellCount,
    static_cast<uint32_t>(y * cellCount * 4),
    rowActive,
    uRightBase, mogrify.x, rowMetric,
    pointScale * mogrify.x,
    depthParameters.w * mogrify.x,
    disparityPrescale,
    maxValidRaw, discontinuityThresholdRaw, cosSquaredThreshold, maxSkippedEdges,
    rowQuadBase, perRowStride,
    outIndices);

  if (rowActive && (threadIdx.x & static_cast<uint32_t>(kGroupWidth - 1)) == 0) {
    if (walked.validCellCount) {
      atomicAdd(&counters->quadCount, walked.quadCount);
      atomicAdd(&counters->stripCount, walked.stripCount);
      atomicAdd(&counters->validCellCount, walked.validCellCount);
    }
  }
}

// ----- Single-thread tail: stamp indirect-draw commands, mirror counters to the host -----

__global__ void stampIndirectArgsKernel(
  const StripMeshCounters* counters,
  RHIDrawIndexedIndirectCommand* outArgs,
  StripMeshCounters* outHostCounters,
  uint32_t reservedQuads) { // deterministic row-contiguous count == trimmedH * trimmedW (0 on empty trim)
  const uint32_t indexCount = reservedQuads * 6u;
  // Slot 0: stereo (2 instances). Slot 1: mono (1 instance).
  outArgs[0] = {indexCount, 2u, 0u, 0, 0u};
  outArgs[1] = {indexCount, 1u, 0u, 0, 0u};
  // Copy counters to host pinned memory for the stats view; the emit path never writes
  // reservedQuadCount (placement is deterministic), so fill it here for the stats mirror.
  *outHostCounters = *counters;
  outHostCounters->reservedQuadCount = reservedQuads;
}

} // namespace

// ----- StripMeshScratch lifecycle -----

void StripMeshScratch::allocate(uint32_t /*W*/, uint32_t /*H*/) {
  if (!d_counters)
    CUDA_CHECK(cuMemAlloc(&d_counters, sizeof(StripMeshCounters)));
  if (!h_counters) {
    CUDA_CHECK(cuMemHostAlloc((void**) &h_counters, sizeof(StripMeshCounters), CU_MEMHOSTALLOC_DEVICEMAP));
    CUDA_CHECK(cuMemHostGetDevicePointer(&h_counters_devicePtr, h_counters, /*flags=*/ 0));
    memset(h_counters, 0, sizeof(StripMeshCounters));
  }
  if (!vertexStream)
    CUDA_CHECK(cuStreamCreate(&vertexStream, CU_STREAM_NON_BLOCKING));
  if (!vertexForkEvent)
    CUDA_CHECK(cuEventCreate(&vertexForkEvent, CU_EVENT_DISABLE_TIMING));
  if (!vertexJoinEvent)
    CUDA_CHECK(cuEventCreate(&vertexJoinEvent, CU_EVENT_DISABLE_TIMING));
}

void StripMeshScratch::destroy() {
  CUDA_SAFE_FREE(d_counters);
  CUDA_SAFE_FREE_HOST(h_counters);
  h_counters_devicePtr = 0;
  if (vertexStream) {
    cuStreamDestroy(vertexStream);
    vertexStream = nullptr;
  }
  if (vertexForkEvent) {
    cuEventDestroy(vertexForkEvent);
    vertexForkEvent = nullptr;
  }
  if (vertexJoinEvent) {
    cuEventDestroy(vertexJoinEvent);
    vertexJoinEvent = nullptr;
  }
}

// ----- Host entrypoint -----

void buildAdaptiveStripMesh(
  const cv::cuda::GpuMat& disparityIn,
  float disparityPrescale,
  glm::vec4 depthParameters,
  glm::mat3 R1,
  glm::vec2 mogrify,
  uint16_t maxValidRaw,
  uint16_t discontinuityThresholdRaw,
  float mergeAngleThresholdRadians,
  uint32_t maxSkippedEdges,
  float pointScale,
  int trimLeft, int trimTop, int trimRight, int trimBottom,
  CUdeviceptr d_vbo,
  CUdeviceptr d_ibo,
  CUdeviceptr d_indirectArgs,
  StripMeshScratch& scratch,
  CUstream stream) {

  assert(disparityIn.type() == CV_16U);
  const int W = disparityIn.cols;
  const int H = disparityIn.rows;

  const int trimmedW = W - (trimLeft + trimRight);
  const int trimmedH = H - (trimTop + trimBottom);

  CUDA_CHECK(cuMemsetD8Async(scratch.d_counters, 0, sizeof(StripMeshCounters), stream));

  if (trimmedW <= 0 || trimmedH <= 0) {
    // Degenerate trim: still stamp (empty) draw commands so the caller's draws are no-ops.
    stampIndirectArgsKernel<<<1, 1, 0, stream>>>(
      reinterpret_cast<const StripMeshCounters*>(scratch.d_counters),
      reinterpret_cast<RHIDrawIndexedIndirectCommand*>(d_indirectArgs),
      reinterpret_cast<StripMeshCounters*>(scratch.h_counters_devicePtr),
      /*reservedQuads=*/ 0u);
    return;
  }

  // Each row reserves a fixed cellCount-quad block, so the drawn index count is
  // deterministic: trimmedH rows * trimmedW quad slots (each row's used prefix + its
  // degenerate-padded tail).
  const uint32_t reservedQuads = static_cast<uint32_t>(trimmedH) * static_cast<uint32_t>(trimmedW);

  // Trimmed view of the disparity mat. step is in bytes (rows may be padded), so the row
  // offset has to go through a byte pointer.
  const PtrStepSz<const uint16_t> trimmedDisparity(
    trimmedH, trimmedW,
    reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(disparityIn.cudaPtr()) + (static_cast<size_t>(trimTop) * disparityIn.step)) + trimLeft,
    disparityIn.step);

  // Matches the point path's texCoordStep = 1 / internalSize; grid position W maps to u = 1.0.
  const glm::vec2 texCoordScale(1.0f / static_cast<float>(W), 1.0f / static_cast<float>(H));

  // The strip walk derives its merge test from the disparity data alone and never reads
  // the VBO, so pass 1 (the vertex bake) runs on the scratch's side stream, concurrent
  // with passes 2-3, and joins the caller's stream before this function returns. The fork
  // event is recorded before pass 2's launch so pass 1 depends only on the caller's prior
  // work (the disparity producer).
  CUDA_CHECK(cuEventRecord(scratch.vertexForkEvent, stream));

  // Pass 2: walk row strips, emitting merged-quad indices straight into the caller's
  // index buffer. 32/kWalkGroupWidth rows per warp. Launched before pass 1: when the GPU
  // is idle, the work distributor drains kernels in arrival order, and pass 1's
  // few-thousand short blocks would occupy every SM slot before this kernel's handful of
  // long-running blocks got resident, serializing the two passes. This kernel's blocks
  // all fit at once, and pass 1 then fills the remaining capacity around them.
  {
    const float cosThreshold = cosf(mergeAngleThresholdRadians);
    constexpr int kRowsPerBlock = kWalkWarpsPerBlock * (32 / kWalkGroupWidth);
    dim3 block(kWalkWarpsPerBlock * 32);
    dim3 grid(divUp(trimmedH, kRowsPerBlock));
    emitStripsKernel<kWalkGroupWidth><<<grid, block, 0, stream>>>(
      trimmedDisparity,
      reinterpret_cast<uint32_t*>(d_ibo),
      reinterpret_cast<StripMeshCounters*>(scratch.d_counters),
      disparityPrescale, depthParameters, mogrify, pointScale,
      /*xOffset=*/ trimLeft, /*yOffset=*/ trimTop,
      maxValidRaw, discontinuityThresholdRaw, cosThreshold * cosThreshold, maxSkippedEdges);
  }

  // Pass 1: populate the vertex buffer, one thread per output vertex (4 per post-trim cell).
  CUDA_CHECK(cuStreamWaitEvent(scratch.vertexStream, scratch.vertexForkEvent, 0));
  {
    dim3 block(128, 1);
    dim3 grid(divUp(trimmedW * 4, block.x), trimmedH);
    populateVertexBufferKernel<<<grid, block, 0, scratch.vertexStream>>>(
      trimmedDisparity,
      disparityPrescale,
      depthParameters,
      R1,
      mogrify,
      pointScale,
      texCoordScale,
      /*xOffset=*/ trimLeft, /*yOffset=*/ trimTop,
      reinterpret_cast<StripMeshVertex*>(d_vbo));
  }
  CUDA_CHECK(cuEventRecord(scratch.vertexJoinEvent, scratch.vertexStream));

  // Pass 3: stamp the indirect draw commands with the deterministic index count and mirror
  // the counters back to the host.
  stampIndirectArgsKernel<<<1, 1, 0, stream>>>(
    reinterpret_cast<const StripMeshCounters*>(scratch.d_counters),
    reinterpret_cast<RHIDrawIndexedIndirectCommand*>(d_indirectArgs),
    reinterpret_cast<StripMeshCounters*>(scratch.h_counters_devicePtr),
    reservedQuads);

  // Join: the caller's stream owns the completed VBO again from here on.
  CUDA_CHECK(cuStreamWaitEvent(stream, scratch.vertexJoinEvent, 0));
}
