// Standalone micro-benchmark / NSight-Compute harness for buildAdaptiveDepthMesh().
//
// Loads a representative 480x270 uint16 disparity surface (packed, headerless) captured
// from the live app, allocates the same worst-case vertex/index/indirect-args buffers the
// real pipeline uses, runs the adaptive-mesh builder in a timed loop, and prints CUDA-event
// timings plus the resulting DepthMeshAdaptiveCounters stats.
//

#include "common/depthMeshAdaptive.h"
#include "rhi/cuda/CudaUtil.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Input geometry is hard-coded per the captured dump (see the bench README/spec).
constexpr int kDisparityWidth = 480;
constexpr int kDisparityHeight = 270;

// Build parameters, mirroring the values the live app feeds buildAdaptiveDepthMesh().
constexpr uint16_t kMaxValidRaw = 128 * 32;
constexpr uint16_t kFlatThresholdRaw = 24;
constexpr uint16_t kDiscontinuityThresholdRaw = 96;
constexpr float kCellOverlapMultiplier = 1.5f;
constexpr int kTrimLeft = 8;
constexpr int kTrimTop = 8;
constexpr int kTrimRight = 8;
constexpr int kTrimBottom = 8;

// The unit under test is invoked many times; a handful of warm-up iterations are excluded
// from the reported statistics so we measure steady-state, not first-launch JIT/alloc cost.
constexpr int kDefaultWarmupIterations = 5;
constexpr int kDefaultTimedIterations = 100;

std::vector<uint16_t> readDisparityRaw(const std::string& path, size_t expectedTexels) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    fprintf(stderr, "Unable to open disparity input \"%s\"\n", path.c_str());
    exit(1);
  }
  const std::streamsize byteCount = file.tellg();
  const size_t expectedBytes = expectedTexels * sizeof(uint16_t);
  if (static_cast<size_t>(byteCount) != expectedBytes) {
    fprintf(stderr, "Disparity input \"%s\" is %lld bytes; expected %zu (%dx%d uint16)\n",
      path.c_str(), static_cast<long long>(byteCount), expectedBytes, kDisparityWidth, kDisparityHeight);
    exit(1);
  }
  file.seekg(0, std::ios::beg);
  std::vector<uint16_t> data(expectedTexels);
  file.read(reinterpret_cast<char*>(data.data()), expectedBytes);
  if (!file) {
    fprintf(stderr, "Short read on disparity input \"%s\"\n", path.c_str());
    exit(1);
  }
  return data;
}

// Resolve the input path: explicit argv[1] wins, otherwise probe the usual run locations
// (the binary may be launched from the repo root or from within the bench directory).
std::string resolveDisparityPath(int argc, char** argv) {
  if (argc > 1)
    return argv[1];
  static const char* kCandidates[] = {
    "disparity.raw",
    "depthMeshAdaptive-bench/disparity.raw",
    "/home/tegra/hmdcam/depthMeshAdaptive-bench/disparity.raw",
  };
  for (const char* candidate : kCandidates) {
    std::ifstream probe(candidate, std::ios::binary);
    if (probe)
      return candidate;
  }
  return kCandidates[0]; // let readDisparityRaw() report the open failure
}

} // namespace

int main(int argc, char** argv) {
  const std::string disparityPath = resolveDisparityPath(argc, argv);
  const int timedIterations = (argc > 2) ? atoi(argv[2]) : kDefaultTimedIterations;

  const int W = kDisparityWidth;
  const int H = kDisparityHeight;
  const size_t texelCount = static_cast<size_t>(W) * static_cast<size_t>(H);

  // ----- CUDA context: retain the device primary context so driver-API allocations here
  // share a context with OpenCV's runtime-API GpuMat allocations. -----
  CUDA_CHECK(cuInit(0));
  CUdevice device = 0;
  CUDA_CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CUDA_CHECK(cuCtxSetCurrent(context));

  char deviceName[256] = {0};
  CUDA_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), device));
  printf("Device: %s\n", deviceName);
  printf("Disparity input: %s (%dx%d uint16)\n", disparityPath.c_str(), W, H);

  // ----- Load and upload the disparity surface. -----
  std::vector<uint16_t> hostDisparity = readDisparityRaw(disparityPath, texelCount);
  cv::Mat hostMat(H, W, CV_16U, hostDisparity.data());
  cv::cuda::GpuMat disparityIn;
  disparityIn.upload(hostMat);

  // ----- Allocate the caller-owned output buffers, sized exactly as DepthMapGenerator.cpp:
  // worst case is one quad per leaf cell (every cell stays at level 0). -----
  const size_t worstCaseQuads = static_cast<size_t>(W) * static_cast<size_t>(H);
  const size_t vboBytes = worstCaseQuads * 4 * sizeof(AdaptiveMeshVertex);
  const size_t iboBytes = worstCaseQuads * 6 * sizeof(uint32_t);
  const size_t indirectArgsBytes = 2 * sizeof(DrawElementsIndirectCommand);

  CUdeviceptr d_vbo = 0, d_ibo = 0, d_indirectArgs = 0;
  CUDA_CHECK(cuMemAlloc(&d_vbo, vboBytes));
  CUDA_CHECK(cuMemAlloc(&d_ibo, iboBytes));
  CUDA_CHECK(cuMemAlloc(&d_indirectArgs, indirectArgsBytes));

  printf("Buffers: vbo=%zu B, ibo=%zu B, indirectArgs=%zu B (worstCaseQuads=%zu)\n",
    vboBytes, iboBytes, indirectArgsBytes, worstCaseQuads);

  DepthMeshAdaptiveScratch scratch;
  scratch.allocate(W, H);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  auto runOnce = [&]() {
    buildAdaptiveDepthMesh(
      disparityIn,
      kMaxValidRaw,
      kFlatThresholdRaw,
      kDiscontinuityThresholdRaw,
      kCellOverlapMultiplier,
      kTrimLeft, kTrimTop, kTrimRight, kTrimBottom,
      d_vbo, d_ibo, d_indirectArgs,
      scratch,
      reinterpret_cast<CUstream>(stream));
  };

  // ----- Warm-up (excluded from timings). -----
  for (int i = 0; i < kDefaultWarmupIterations; ++i)
    runOnce();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // ----- Timed loop: one CUDA-event pair per iteration for min/avg/max. -----
  cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));

  double totalMs = 0.0;
  double minMs = 1e30;
  double maxMs = 0.0;
  for (int i = 0; i < timedIterations; ++i) {
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    runOnce();
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    totalMs += ms;
    if (ms < minMs) minMs = ms;
    if (ms > maxMs) maxMs = ms;
  }

  // Stats live in device-mapped pinned host memory written on the stream; valid post-sync.
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const DepthMeshAdaptiveCounters& counters = *scratch.h_counters;

  // Order-independent checksum of the vertex multiset: quads are emitted in nondeterministic
  // (atomic) order, so we hash each vertex on its own and sum the hashes commutatively. This
  // matches iff the same set of vertices was produced -- proving a refactor bit-identical in
  // the welded disparities and positions, independent of emission order. (The index pattern
  // per quad is structurally fixed, so the vertex multiset pins down the mesh.)
  auto fnv1a = [](const void* data, size_t bytes) -> uint64_t {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
      h ^= p[i];
      h *= 1099511628211ull;
    }
    return h;
  };
  std::vector<AdaptiveMeshVertex> vboHost(counters.vertexCounter);
  CUDA_CHECK(cuMemcpyDtoH(vboHost.data(), d_vbo, vboHost.size() * sizeof(AdaptiveMeshVertex)));
  uint64_t vertexSetHash = 0;
  for (const AdaptiveMeshVertex& v : vboHost)
    vertexSetHash += fnv1a(&v, sizeof(AdaptiveMeshVertex));
  printf("\n=== Output checksum ===\n");
  printf("  vertex-set (order-independent): %016llx\n", (unsigned long long) vertexSetHash);

  // ----- Report. -----
  printf("\n=== Timing over %d iterations (warm-up %d excluded) ===\n",
    timedIterations, kDefaultWarmupIterations);
  printf("  avg: %.4f ms   min: %.4f ms   max: %.4f ms\n",
    totalMs / static_cast<double>(timedIterations), minMs, maxMs);

  printf("\n=== DepthMeshAdaptiveCounters ===\n");
  printf("  anchors:  %u  (work-list entries; should equal quads)\n", counters.anchorCount);
  printf("  vertices: %u  (quads: %u)\n", counters.vertexCounter, counters.vertexCounter / 4u);
  printf("  indices:  %u  (triangles: %u)\n", counters.indexCounter, counters.indexCounter / 3u);
  printf("  level histogram (cells emitted per pyramid level):\n");
  uint32_t totalCells = 0;
  for (int L = 0; L < kAdaptiveMeshLevels; ++L)
    totalCells += counters.levelHistograms[L];
  for (int L = 0; L < kAdaptiveMeshLevels; ++L) {
    const uint32_t count = counters.levelHistograms[L];
    const double pct = totalCells ? (100.0 * count / totalCells) : 0.0;
    printf("    L%d (%dx%d cells): %8u  (%.1f%%)\n", L, 1 << L, 1 << L, count, pct);
  }
  printf("    total emitted cells: %u\n", totalCells);

  // ----- Cleanup. -----
  CUDA_CHECK(cudaEventDestroy(startEvent));
  CUDA_CHECK(cudaEventDestroy(stopEvent));
  CUDA_CHECK(cudaStreamDestroy(stream));
  scratch.destroy();
  CUDA_SAFE_FREE(d_vbo);
  CUDA_SAFE_FREE(d_ibo);
  CUDA_SAFE_FREE(d_indirectArgs);
  cuDevicePrimaryCtxRelease(device);
  return 0;
}
