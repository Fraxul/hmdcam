#pragma once
#include <cuda.h>
#include <opencv2/core/cuda.hpp>
#include <stdint.h>

// ABI of glDrawElementsIndirect's command buffer (20 bytes).
struct DrawElementsIndirectCommand {
  uint32_t count;
  uint32_t instanceCount;
  uint32_t firstIndex;
  int32_t baseVertex;
  uint32_t baseInstance;
};

// Number of fractional bits used to encode gridX/gridY -- positions are stored as
// pixel * (1 << kAdaptiveMeshGridFracBits) so sub-pixel offsets (used for partial-cell
// overlap at cracked edges) survive the uint16_t serialization. With 4 fractional bits
// the representable range is up to ~4095 pixels in each axis, plenty for the disparity
// textures this pipeline runs on. The vertex shader divides by the same scale.
constexpr int kAdaptiveMeshGridFracBits = 4;
constexpr int kAdaptiveMeshGridScale = 1 << kAdaptiveMeshGridFracBits;

// Set to 1 to compile the per-cell level-color debug visualization. This widens every
// vertex by 4 bytes (an extra level/snap-flags word) and enables the LEVEL_DEBUG shader
// path, gated at runtime by the "Debug: Adaptive level color" checkbox. Off in production:
// zero vertex-size and shader cost. The macro lives here (the one header shared by the CUDA
// builder and the C++ vertex-layout setup) so both translation units agree on the struct.
#ifndef ADAPTIVE_MESH_DEBUG
#define ADAPTIVE_MESH_DEBUG 0
#endif

// Per-vertex format emitted by the adaptive mesh builder. Matches the GL vertex layout
// declared for the m_disparityDepthMapAdaptivePipeline (see DepthMapGenerator.cpp).
// gridX/gridY are q12.4 fixed-point pixel coordinates (pixel * kAdaptiveMeshGridScale).
struct AdaptiveMeshVertex {
  uint16_t gridX;
  uint16_t gridY;
  float disparityRaw;
#if ADAPTIVE_MESH_DEBUG
  // Flat per-cell diagnostic word: bits 0-3 = chosen pyramid level L, bit 4 = right edge
  // snapped to dRep, bit 5 = bottom edge snapped to dRep. All four corners of a cell carry
  // the same value. Only present (and only read by the shader) when ADAPTIVE_MESH_DEBUG.
  uint16_t debugLevelFlags;
#endif
};
#if ADAPTIVE_MESH_DEBUG
static_assert(sizeof(AdaptiveMeshVertex) == 12, "AdaptiveMeshVertex must be 12 bytes in debug");
#else
static_assert(sizeof(AdaptiveMeshVertex) == 8, "AdaptiveMeshVertex must be 8 bytes");
#endif

// Bit layout of AdaptiveMeshVertex::debugLevelFlags.
constexpr uint16_t kAdaptiveDebugLevelMask = 0x000F;
constexpr uint16_t kAdaptiveDebugRightSnap = 0x0010;
constexpr uint16_t kAdaptiveDebugBottomSnap = 0x0020;
constexpr uint16_t kAdaptiveDebugTopSnap = 0x0040;
constexpr uint16_t kAdaptiveDebugLeftSnap = 0x0080;
constexpr uint16_t kAdaptiveDebugSnapMask = 0x00F0; // any edge snapped

// Number of pyramid levels. Level 0 = 1x1 cells, level kAdaptiveMeshLevels-1 = the largest merge size.
constexpr int kAdaptiveMeshLevels = 5;
constexpr int kAdaptiveMeshMaxBlockSize = 1 << (kAdaptiveMeshLevels - 1);

// Structure shared between host and device.
// Device kernels use it with atomics for bookkeeping,
// then it's copied to the host at the end for the stats view.
struct DepthMeshAdaptiveCounters {
  uint32_t vertexCounter; // total verts emitted; quad count = this / 4
  uint32_t indexCounter;
  uint32_t levelHistograms[kAdaptiveMeshLevels];
};

struct DepthMeshAdaptiveScratch {
  // mip[L] holds one packed (mn, mx) cell per (W>>L)x(H>>L) block, type CV_32S.
  // INVALID_CELL sentinel (0xFFFFFFFF) marks blocks that contain any invalid sample.
  cv::cuda::GpuMat mip[kAdaptiveMeshLevels];

  // Per-leaf-cell decision: 0 = skip emission, otherwise the chosen level + 1.
  cv::cuda::GpuMat maxFlatLevel; // CV_8U, WxH

  // Device-side counters and per-level histogram
  CUdeviceptr d_counters = 0; // struct DepthMeshAdaptiveCounters

  // Host-side mirror of the most recently completed frame's stats (for IMGUI debug).
  // Updated by writeIndirectArgsKernel at the end of the process.
  DepthMeshAdaptiveCounters* h_counters = nullptr;
  CUdeviceptr h_counters_devicePtr = 0;

  void allocate(uint32_t W, uint32_t H);
  void destroy();

  ~DepthMeshAdaptiveScratch() { destroy(); }
};

// Build an adaptive triangle mesh from disparityIn into a vertex/index buffer pair.
//
// Inputs:
//   disparityIn       - the post-processed disparity (uint16_t per texel).
//   maxValidRaw       - inclusive maximum valid disparity in raw units.
//   flatThresholdRaw  - inclusive max (max - min) within a block that still counts as flat.
//   discontinuityThresholdRaw - max disparity step across a quad edge that still gets
//                       welded. Each emitted cell uses its top-left texel as its
//                       representative disparity; any corner whose welded value (the
//                       corner texel, or a T-junction lerp along a coarser neighbor's
//                       edge) differs from the representative by more than this
//                       threshold reverts to the representative, leaving an intentional
//                       crack between this cell and the neighbor on that edge.
//                       At threshold = 0 every edge is discontinuous and the mesh is
//                       a set of flat, disconnected quads; at the max value every edge
//                       welds and the mesh is fully connected.
//   cellOverlapMultiplier - on a cracked edge, the far-side cell (lower disparity)
//                       extends its quad outward past the natural cell boundary by
//                       ceil((multiplier - 1) * cellSize) pixels, so the back surface
//                       overlaps into the crack and hides the gap from off-axis
//                       viewpoints. 1.0 disables overlap entirely; 1.5 = 50% extension.
//   trimLeft/Top/Right/Bottom - cells in the trimmed border are treated as invalid.
//
// Outputs (caller-owned, all device pointers; buffers must be sized for worst-case
// W*H quads = 4*W*H verts and 6*W*H indices):
//   d_vbo            - AdaptiveMeshVertex[]
//   d_ibo            - uint32_t[]
//   d_indirectArgs   - DrawElementsIndirectCommand[2]; slot 0 has instanceCount=2 (stereo),
//                      slot 1 has instanceCount=1 (mono). Both share the same index count.
//
// The function records work onto stream and updates scratch.h_levelHistogram /
// scratch.h_quadCount asynchronously (host data is valid after stream completion).
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
  CUstream stream);
