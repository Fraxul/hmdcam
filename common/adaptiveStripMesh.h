#pragma once
#include "rhi/RHIIndirectDrawCommand.h"
#include <cuda.h>
#include <glm/glm.hpp>
#include <opencv2/core/cuda.hpp>
#include <stdint.h>

// Vertex buffer format. 16 bytes so the builder can write each vertex as a single
// float4 store; the alignas guarantees that store is aligned.
struct alignas(16) StripMeshVertex {
  // Precomputed local-space position (the output of the point shader's TransformToLocalSpace).
  glm::vec3 localPosition;

  // Rectified texture coordinates, pre-distortion, unorm16-packed. The vertex shader
  // unpacks these and applies the colorReprojection homography + distortion map lookup.
  uint16_t texCoords[2];
};
static_assert(sizeof(StripMeshVertex) == 16);

// Structure shared between host and device. Device kernels accumulate with atomics;
// the tail kernel copies it to host pinned memory for the stats view.
struct StripMeshCounters {
  uint32_t quadCount; // total merged (non-degenerate) quads emitted
  uint32_t stripCount; // welded runs; a discontinuity or invalid gap starts a new strip
  uint32_t validCellCount; // cells passing the validity test == quad count of the equivalent point mesh
  uint32_t reservedQuadCount; // chunk-granular index-buffer allocation cursor; index count = reservedQuadCount * 6
};

// The strip walk writes quads directly into the caller's index buffer through a global
// cursor, reserving space in fixed-size chunks (one atomic per chunk instead of per quad,
// prefetched one chunk ahead to keep the atomic's latency off the walk's critical path).
// Each row pads its final partial chunk and its one outstanding prefetched chunk with
// degenerate quads (all six indices equal; zero-area, culled by the rasterizer), so the
// worst case exceeds one quad per cell by (2 * chunk - 1) per row. Size d_ibo for
// stripMeshWorstCaseQuads(W, H) * 6 indices.
constexpr uint32_t kStripMeshEmitChunkQuads = 8;
constexpr size_t stripMeshWorstCaseQuads(uint32_t W, uint32_t H) {
  return (static_cast<size_t>(W) * H) + (static_cast<size_t>(H) * ((2 * kStripMeshEmitChunkQuads) - 1));
}

struct StripMeshScratch {
  // Device-side counters, reset at the start of each build.
  CUdeviceptr d_counters = 0; // StripMeshCounters

  // Host-side mirror of the most recently completed frame's stats (for IMGUI debug).
  // Written by the arg-stamp kernel; valid after stream completion.
  StripMeshCounters* h_counters = nullptr;
  CUdeviceptr h_counters_devicePtr = 0;

  // Side stream for the vertex-buffer bake: the strip walk derives its merge test from
  // the disparity data alone, so pass 1 runs concurrently with passes 2-3 and joins the
  // caller's stream before buildAdaptiveStripMesh returns. Non-blocking, so this works
  // (and still overlaps) even when the caller passes the legacy default stream.
  CUstream vertexStream = nullptr;
  CUevent vertexForkEvent = nullptr;
  CUevent vertexJoinEvent = nullptr;

  void allocate(uint32_t W, uint32_t H);
  void destroy();

  ~StripMeshScratch() { destroy(); }
};

// Build an adaptive strip mesh from disparityIn into a vertex/index buffer pair.
//
// Pass 1 precomputes the point-rendering quad corners: four local-space vertices per
// disparity cell, exactly what meshDisparityDepthMapPoints.vtx.glsl would generate
// (including the pointScale quad expansion). Pass 2 walks each row and greedily merges
// runs of near-collinear quads, emitting only the indices for the surviving edges:
// a quad is extended across the next cell while the angle between (leading edge ->
// current trailing edge) and (leading edge -> candidate trailing edge) stays under
// mergeAngleThresholdRadians, up to maxSkippedEdges internal edges per quad. When a
// quad ends for curvature reasons the next quad reuses its trailing-edge vertices, so
// strips stay watertight; a raw-disparity step above discontinuityThresholdRaw (or an
// invalid cell) breaks the strip instead, leaving an intentional crack that the
// pointScale overlap covers, same as the point path.
//
// Inputs:
//   disparityIn        - post-processed disparity, CV_16U.
//   disparityPrescale  - raw units -> disparity pixels (fold debugDisparityScale in here).
//   depthParameters    - CameraSystem::View::depthParameters(), as in the uniform block.
//   R1                 - rectification rotation (upper-left 3x3 of the uniform block's R1).
//   mogrify            - algorithm downsample factors (m_algoDownsampleX / Y).
//   maxValidRaw        - inclusive maximum valid disparity in raw units.
//   discontinuityThresholdRaw - max raw-disparity step between adjacent cells that still
//                        welds; larger steps break the strip. 0xFFFF never breaks.
//   mergeAngleThresholdRadians - max angular deviation before a merged quad is ended.
//                        Must be < pi/2 (the test assumes an acute angle).
//   maxSkippedEdges    - internal-edge skip limit per merged quad (a quad spans at most
//                        maxSkippedEdges + 1 cells).
//   pointScale         - quad expansion factor, same semantics as the point path.
//   trimLeft/Top/Right/Bottom - border cells excluded from the mesh entirely.
//
// Outputs (caller-owned device pointers):
//   d_vbo            - StripMeshVertex[W * H * 4], written densely over the trimmed grid.
//   d_ibo            - uint32_t[stripMeshWorstCaseQuads(W, H) * 6]; indices reference
//                      d_vbo with vertexOffset 0. Each row emits into its own contiguous,
//                      x-ordered block (rowIndex * trimmedW quads), used prefix first and
//                      the unused tail padded with degenerate quads -- welded neighbors stay
//                      adjacent for the rasterizer's post-transform vertex-reuse window. The
//                      draw covers trimmedW * trimmedH quads (the padding is culled).
//   d_indirectArgs   - RHIDrawIndexedIndirectCommand[2]; slot 0 has instanceCount=2
//                      (stereo), slot 1 has instanceCount=1 (mono). Matches the
//                      adaptive-mesh path so the draw code is shared.
//
// The function records work onto stream; scratch.h_counters is valid after stream completion.
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
  CUstream stream);
