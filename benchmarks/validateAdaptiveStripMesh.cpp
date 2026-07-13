// Validation + micro-benchmark harness for buildAdaptiveStripMesh().
//
// Feeds synthetic disparity patterns (constant / ramp / step / invalid gaps) plus the real
// captured disparity.raw from depthMeshAdaptive-bench, then checks structural invariants of
// the emitted mesh:
//   - counters vs indirect args consistency
//   - quad index encoding (lead/trail vertex pattern, winding layout)
//   - index bounds, same-row constraint, merge-span cap
//   - exact tiling: per row, emitted quads cover exactly the valid cells, no gaps/overlaps
//   - weld continuity: within a strip, each quad starts at the previous quad's trailing verts
//   - vertex positions/texcoords vs a CPU reference of the TransformToLocalSpace math

#include "common/adaptiveStripMesh.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/core/cuda.hpp>
#include <nvtx3/nvToolsExt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "rhi/cuda/CudaUtil.h"
#define CHECK_CU(x) CUDA_CHECK(x)

namespace {

int g_failures = 0;
#define EXPECT(cond, ...)                 \
  do {                                    \
    if (!(cond)) {                        \
      fprintf(stderr, "FAIL: ");          \
      fprintf(stderr, __VA_ARGS__);       \
      fprintf(stderr, "  [%s]\n", #cond); \
      ++g_failures;                       \
    }                                     \
  } while (0)

struct BuildParams {
  float disparityPrescale = 1.0f / 32.0f;
  glm::vec4 depthParameters = glm::vec4(-960.0f, -540.0f, 1000.0f, 15.6f);
  glm::mat3 R1 = glm::mat3(1.0f);
  glm::vec2 mogrify = glm::vec2(4.0f, 4.0f);
  uint16_t maxValidRaw = 4096;
  uint16_t discontinuityThresholdRaw = 96;
  float mergeAngleRadians = 2.0f * (float) M_PI / 180.0f;
  uint32_t maxSkippedEdges = 4;
  float pointScale = 1.0f;
  int trimLeft = 8, trimTop = 8, trimRight = 8, trimBottom = 8;
};

struct BuildOutputs {
  StripMeshCounters counters;
  RHIDrawIndexedIndirectCommand args[2];
  std::vector<StripMeshVertex> vbo; // trimmedW*trimmedH*4, dense trimmed layout
  std::vector<uint32_t> ibo; // counters.reservedQuadCount*6 entries: real quads in
    // cursor-allocation order plus degenerate row-padding fillers
};

BuildOutputs runBuild(const std::vector<uint16_t>& hostDisparity, int W, int H, const BuildParams& p) {
  cv::Mat hostMat(H, W, CV_16U, const_cast<uint16_t*>(hostDisparity.data()));
  cv::cuda::GpuMat disparityIn;
  disparityIn.upload(hostMat);

  const size_t cellCount = size_t(W) * size_t(H);
  const size_t iboIndexCapacity = stripMeshWorstCaseQuads(uint32_t(W), uint32_t(H)) * 6;
  CUdeviceptr d_vbo = 0, d_ibo = 0, d_args = 0;
  CHECK_CU(cuMemAlloc(&d_vbo, cellCount * 4 * sizeof(StripMeshVertex)));
  CHECK_CU(cuMemAlloc(&d_ibo, iboIndexCapacity * sizeof(uint32_t)));
  CHECK_CU(cuMemAlloc(&d_args, 2 * sizeof(RHIDrawIndexedIndirectCommand)));
  // Poison the IBO so stale data can't masquerade as valid indices.
  CHECK_CU(cuMemsetD32(d_ibo, 0xDEADBEEF, iboIndexCapacity));

  StripMeshScratch scratch;
  scratch.allocate(uint32_t(W), uint32_t(H));

  buildAdaptiveStripMesh(
    disparityIn,
    p.disparityPrescale, p.depthParameters, p.R1, p.mogrify,
    p.maxValidRaw, p.discontinuityThresholdRaw, p.mergeAngleRadians, p.maxSkippedEdges,
    p.pointScale,
    p.trimLeft, p.trimTop, p.trimRight, p.trimBottom,
    d_vbo, d_ibo, d_args, scratch, /*stream=*/ 0);
  CHECK_CU(cuStreamSynchronize(0));

  const int tw = W - p.trimLeft - p.trimRight;
  const int th = H - p.trimTop - p.trimBottom;

  BuildOutputs out;
  out.counters = *scratch.h_counters;
  CHECK_CU(cuMemcpyDtoH(out.args, d_args, sizeof(out.args)));
  out.vbo.resize(size_t(tw) * size_t(th) * 4);
  CHECK_CU(cuMemcpyDtoH(out.vbo.data(), d_vbo, out.vbo.size() * sizeof(StripMeshVertex)));
  out.ibo.resize(size_t(out.counters.reservedQuadCount) * 6);
  if (!out.ibo.empty())
    CHECK_CU(cuMemcpyDtoH(out.ibo.data(), d_ibo, out.ibo.size() * sizeof(uint32_t)));

  cuMemFree(d_vbo);
  cuMemFree(d_ibo);
  cuMemFree(d_args);
  return out;
}

// CPU reference of the vertex math.
glm::vec3 referencePosition(float gx, float gy, float rawDisparity, const BuildParams& p) {
  float fDisp = std::max(rawDisparity * p.disparityPrescale, 1.0f / 32.0f);
  float lw = p.depthParameters.w * (fDisp * p.mogrify.x);
  glm::vec3 pp(gx * p.mogrify.x + p.depthParameters.x, gy * p.mogrify.y + p.depthParameters.y, p.depthParameters.z);
  return p.R1 * (pp / lw);
}

void validate(const char* label, const std::vector<uint16_t>& disp, int W, int H, const BuildParams& p, bool verbose = true) {
  BuildOutputs out = runBuild(disp, W, H, p);
  const int tw = W - p.trimLeft - p.trimRight;
  const int th = H - p.trimTop - p.trimBottom;
  const uint32_t vertexCount = uint32_t(tw) * uint32_t(th) * 4;

  // Counters vs indirect args. The draw covers the chunk-granular reserved range; real
  // quads plus per-row padding (at most one partial chunk of waste per row).
  EXPECT(out.args[0].indexCount == out.counters.reservedQuadCount * 6, "%s: stereo indexCount %u != reserved quads %u * 6\n", label, out.args[0].indexCount, out.counters.reservedQuadCount);
  EXPECT(out.args[1].indexCount == out.counters.reservedQuadCount * 6, "%s: mono indexCount mismatch\n", label);
  EXPECT(out.counters.reservedQuadCount % kStripMeshEmitChunkQuads == 0, "%s: reserved %u not chunk-granular\n", label, out.counters.reservedQuadCount);
  EXPECT(out.counters.reservedQuadCount >= out.counters.quadCount, "%s: reserved %u < quads %u\n", label, out.counters.reservedQuadCount, out.counters.quadCount);
  EXPECT(out.counters.reservedQuadCount <= out.counters.quadCount + uint32_t(th) * ((2 * kStripMeshEmitChunkQuads) - 1), "%s: reserved %u exceeds padding bound (quads %u)\n", label, out.counters.reservedQuadCount, out.counters.quadCount);
  EXPECT(out.args[0].instanceCount == 2 && out.args[1].instanceCount == 1, "%s: instance counts\n", label);
  EXPECT(out.args[0].baseIndex == 0 && out.args[0].vertexOffset == 0 && out.args[0].baseInstance == 0, "%s: stereo cmd offsets\n", label);

  // Ground-truth valid cell count.
  uint32_t expectValid = 0;
  for (int y = 0; y < th; ++y)
    for (int x = 0; x < tw; ++x)
      if (disp[size_t(y + p.trimTop) * W + (x + p.trimLeft)] <= p.maxValidRaw) ++expectValid;
  EXPECT(out.counters.validCellCount == expectValid, "%s: validCellCount %u != expected %u\n", label, out.counters.validCellCount, expectValid);

  // Per-quad structural checks + coverage tally. A row's real quads appear in walk order
  // (its chunk reservations are sequential, so their slots increase monotonically) even
  // though different rows' chunks interleave; degenerate row-padding fillers (all six
  // indices equal) are skipped.
  std::vector<uint8_t> covered(size_t(tw) * size_t(th), 0);
  uint32_t realQuads = 0;
  // Per-row quads as (firstCell, lastCell, leadVert, trailVert) for weld/order checks.
  std::map<int, std::vector<std::array<uint32_t, 4>>> rowQuads;
  for (uint32_t q = 0; q < out.counters.reservedQuadCount; ++q) {
    const uint32_t* ix = &out.ibo[q * 6];
    if (ix[0] == ix[1]) { // filler
      EXPECT(ix[2] == ix[0] && ix[3] == ix[0] && ix[4] == ix[0] && ix[5] == ix[0], "%s: filler quad %u not fully degenerate\n", label, q);
      continue;
    }
    ++realQuads;
    uint32_t lead = ix[0], trail = ix[2];
    EXPECT(ix[1] == lead + 1 && ix[3] == trail && ix[4] == lead + 1 && ix[5] == trail + 1, "%s: quad %u index pattern\n", label, q);
    EXPECT(lead < vertexCount && trail + 1 < vertexCount, "%s: quad %u index out of range\n", label, q);
    EXPECT((trail % 4) == 2, "%s: quad %u trail corner %u\n", label, q, trail % 4);
    uint32_t leadCorner = lead % 4;
    EXPECT(leadCorner == 0 || leadCorner == 2, "%s: quad %u lead corner %u\n", label, q, leadCorner);
    uint32_t leadCell = lead / 4, trailCell = trail / 4;
    int rowL = int(leadCell) / tw, rowT = int(trailCell) / tw;
    EXPECT(rowL == rowT, "%s: quad %u spans rows %d/%d\n", label, q, rowL, rowT);
    // Cells covered: a right-corner lead means the quad starts at the next cell.
    uint32_t firstCell = leadCell + (leadCorner == 2 ? 1 : 0);
    EXPECT(firstCell <= trailCell, "%s: quad %u firstCell %u > trailCell %u\n", label, q, firstCell, trailCell);
    EXPECT(trailCell - firstCell + 1 <= p.maxSkippedEdges + 1, "%s: quad %u span %u exceeds cap\n", label, q, trailCell - firstCell + 1);
    for (uint32_t c = firstCell; c <= trailCell; ++c) {
      EXPECT(!covered[c], "%s: cell %u covered twice\n", label, c);
      covered[c] = 1;
    }
    rowQuads[rowL].push_back({firstCell, trailCell, lead, trail});
  }
  EXPECT(realQuads == out.counters.quadCount, "%s: %u real quads in ibo != counter %u\n", label, realQuads, out.counters.quadCount);

  // Exact tiling of valid cells.
  for (int y = 0; y < th; ++y) {
    for (int x = 0; x < tw; ++x) {
      bool valid = disp[size_t(y + p.trimTop) * W + (x + p.trimLeft)] <= p.maxValidRaw;
      if (covered[size_t(y) * tw + x] != (valid ? 1 : 0)) {
        EXPECT(false, "%s: coverage mismatch at (%d,%d): valid=%d covered=%d\n", label, x, y, valid ? 1 : 0, covered[size_t(y) * tw + x]);
        goto tilingDone; // one report is enough
      }
    }
  }
tilingDone:

  // Weld continuity: within each row, quads land contiguously in the IBO in walk order,
  // so a quad that isn't a fresh strip start must reuse its predecessor's trailing verts.
  uint32_t weldedJoints = 0, strips = 0;
  for (auto& kv : rowQuads) {
    auto& quads = kv.second;
    for (size_t i = 0; i < quads.size(); ++i) {
      bool freshStart = (quads[i][2] % 4) == 0;
      if (i == 0) {
        EXPECT(freshStart, "%s: row %d first quad not a fresh strip start\n", label, kv.first);
      } else if (!freshStart) {
        EXPECT(quads[i][2] == quads[i - 1][3], "%s: row %d quad %zu weld broken (lead %u vs prev trail %u)\n", label, kv.first, i, quads[i][2], quads[i - 1][3]);
        ++weldedJoints;
      }
      if (freshStart) ++strips;
    }
  }
  EXPECT(strips == out.counters.stripCount, "%s: stripCount %u != observed %u\n", label, out.counters.stripCount, strips);

  // Vertex spot checks vs CPU reference (corner positions + texcoords), sampled grid.
  int checked = 0;
  for (int y = 0; y < th; y += 7) {
    for (int x = 0; x < tw; x += 13) {
      float raw = float(disp[size_t(y + p.trimTop) * W + (x + p.trimLeft)]);
      const StripMeshVertex& v0 = out.vbo[(size_t(y) * tw + x) * 4 + 0];
      const StripMeshVertex& v3 = out.vbo[(size_t(y) * tw + x) * 4 + 3];
      glm::vec3 r0 = referencePosition(float(x + p.trimLeft), float(y + p.trimTop), raw, p);
      glm::vec3 r3 = referencePosition(float(x + p.trimLeft) + p.pointScale, float(y + p.trimTop) + p.pointScale, raw, p);
      float e0 = glm::length(v0.localPosition - r0) / std::max(1.0f, glm::length(r0));
      float e3 = glm::length(v3.localPosition - r3) / std::max(1.0f, glm::length(r3));
      EXPECT(e0 < 1e-4f && e3 < 1e-4f, "%s: vertex (%d,%d) position error %g/%g\n", label, x, y, e0, e3);
      float expectU0 = std::min(1.0f, float(x + p.trimLeft) / float(W));
      float expectV3 = std::min(1.0f, (float(y + p.trimTop) + p.pointScale) / float(H));
      EXPECT(std::abs(v0.texCoords[0] / 65535.0f - expectU0) < 1e-4f, "%s: vertex (%d,%d) texcoord u\n", label, x, y);
      EXPECT(std::abs(v3.texCoords[1] / 65535.0f - expectV3) < 1e-4f, "%s: vertex (%d,%d) texcoord v\n", label, x, y);
      ++checked;
    }
  }

  if (verbose) {
    printf("%-28s cells=%6u quads=%6u strips=%5u  geometry=%5.1f%% of point mesh  (verts checked: %d)\n",
      label, out.counters.validCellCount, out.counters.quadCount, out.counters.stripCount,
      out.counters.validCellCount ? 100.0f * float(out.counters.quadCount) / float(out.counters.validCellCount) : 0.0f,
      checked);
  }
}

std::vector<uint16_t> loadRaw(const char* path, size_t texels) {
  std::ifstream f(path, std::ios::binary);
  std::vector<uint16_t> data(texels);
  f.read(reinterpret_cast<char*>(data.data()), texels * sizeof(uint16_t));
  if (!f) {
    fprintf(stderr, "failed to read %s\n", path);
    exit(1);
  }
  return data;
}

} // namespace

int main() {
  nvtxInitialize(nullptr);
  CHECK_CU(cuInit(0));
  CUdevice device = 0;
  CHECK_CU(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
  CHECK_CU(cuDevicePrimaryCtxRetain(&context, device));
  CHECK_CU(cuCtxSetCurrent(context));

  const int W = 64, H = 16;
  BuildParams p;
  p.trimLeft = 2;
  p.trimTop = 1;
  p.trimRight = 2;
  p.trimBottom = 1;

  { // Constant disparity: expect maximal merging (all quads at the span cap).
    std::vector<uint16_t> disp(size_t(W) * H, 800);
    validate("constant", disp, W, H, p);
    BuildOutputs out = runBuild(disp, W, H, p);
    // 60 cells/row -> 12 quads of 5 cells each.
    EXPECT(out.counters.quadCount == out.counters.validCellCount / (p.maxSkippedEdges + 1), "constant: expected full merging (%u quads for %u cells)\n", out.counters.quadCount, out.counters.validCellCount);
    EXPECT(out.counters.stripCount == uint32_t(H - 2), "constant: expected 1 strip per row, got %u\n", out.counters.stripCount);
  }

  { // Horizontal ramp: locally collinear in disparity space -> should still merge heavily.
    std::vector<uint16_t> disp(size_t(W) * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        disp[size_t(y) * W + x] = uint16_t(600 + x * 8);
    validate("ramp", disp, W, H, p);
  }

  { // Step discontinuity at x=32: must crack, never bridge.
    std::vector<uint16_t> disp(size_t(W) * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        disp[size_t(y) * W + x] = (x < 32) ? 800 : 2400;
    validate("step", disp, W, H, p);
    BuildOutputs out = runBuild(disp, W, H, p);
    EXPECT(out.counters.stripCount == uint32_t(2 * (H - 2)), "step: expected 2 strips per row, got %u\n", out.counters.stripCount);
    // No quad's cells may span the step column.
    for (uint32_t q = 0; q < out.counters.reservedQuadCount; ++q) {
      uint32_t lead = out.ibo[q * 6], trail = out.ibo[q * 6 + 2];
      if (lead == out.ibo[q * 6 + 1]) continue; // filler
      uint32_t firstCell = (lead / 4) % uint32_t(W - 4) + ((lead % 4) == 2 ? 1 : 0);
      uint32_t lastCell = (trail / 4) % uint32_t(W - 4);
      // step column in trimmed coords: 32 - trimLeft = 30
      EXPECT(!(firstCell < 30 && lastCell >= 30), "step: quad %u bridges the discontinuity (%u..%u)\n", q, firstCell, lastCell);
    }
  }

  { // Invalid gaps: columns 20..27 invalid, plus scattered invalid speckles.
    std::vector<uint16_t> disp(size_t(W) * H, 1000);
    for (int y = 0; y < H; ++y) {
      for (int x = 20; x < 28; ++x) disp[size_t(y) * W + x] = 0xFFFF;
      disp[size_t(y) * W + ((y * 7) % W)] = 0xFFFF;
    }
    validate("invalid-gaps", disp, W, H, p);
  }

  { // pointScale > 1 exercises the overlap path and texcoord clamping.
    BuildParams ps = p;
    ps.pointScale = 1.5f;
    std::vector<uint16_t> disp(size_t(W) * H, 800);
    validate("constant-pointScale1.5", disp, W, H, ps);
  }

  { // Non-identity R1 (small rotation about Y) to make sure the rotation path is exercised.
    BuildParams pr = p;
    float a = 0.05f;
    pr.R1 = glm::mat3(cosf(a), 0, -sinf(a), 0, 1, 0, sinf(a), 0, cosf(a));
    std::vector<uint16_t> disp(size_t(W) * H);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        disp[size_t(y) * W + x] = uint16_t(600 + ((x * 13 + y * 29) % 40));
    validate("noisy-R1", disp, W, H, pr);
  }

  { // Real captured disparity at several merge-angle thresholds.
    const int RW = 480, RH = 270;
    std::vector<uint16_t> disp = loadRaw("/home/tegra/hmdcam/depthMeshAdaptive-bench/disparity.raw", size_t(RW) * RH);
    BuildParams pReal; // bench-style trim of 8 all around, defaults otherwise
    for (float degrees : {0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 60.0f}) {
      pReal.mergeAngleRadians = degrees * (float) M_PI / 180.0f;
      char label[64];
      snprintf(label, sizeof(label), "real capture @ %.1f deg", degrees);
      validate(label, disp, RW, RH, pReal);
    }
  }

  { // Timing on the real capture (steady state, 100 iterations).
    nvtxRangePushA("Benchmark");
    const int RW = 480, RH = 270;
    std::vector<uint16_t> disp = loadRaw("/home/tegra/hmdcam/depthMeshAdaptive-bench/disparity.raw", size_t(RW) * RH);
    cv::Mat hostMat(RH, RW, CV_16U, disp.data());
    cv::cuda::GpuMat disparityIn;
    disparityIn.upload(hostMat);
    BuildParams pReal;
    const size_t cellCount = size_t(RW) * RH;
    CUdeviceptr d_vbo = 0, d_ibo = 0, d_args = 0;
    CHECK_CU(cuMemAlloc(&d_vbo, cellCount * 4 * sizeof(StripMeshVertex)));
    CHECK_CU(cuMemAlloc(&d_ibo, cellCount * 6 * sizeof(uint32_t)));
    CHECK_CU(cuMemAlloc(&d_args, 2 * sizeof(RHIDrawIndexedIndirectCommand)));
    StripMeshScratch scratch;
    scratch.allocate(uint32_t(RW), uint32_t(RH));
    auto once = [&]() {
      buildAdaptiveStripMesh(disparityIn, pReal.disparityPrescale, pReal.depthParameters, pReal.R1, pReal.mogrify,
        pReal.maxValidRaw, pReal.discontinuityThresholdRaw, pReal.mergeAngleRadians, pReal.maxSkippedEdges, pReal.pointScale,
        pReal.trimLeft, pReal.trimTop, pReal.trimRight, pReal.trimBottom, d_vbo, d_ibo, d_args, scratch, 0);
    };
    for (int i = 0; i < 10; ++i) once();
    CHECK_CU(cuStreamSynchronize(0));
    // Synchronize per iteration: a deeply-queued loop measures artifacts of the harness's
    // launch pattern, not per-frame cost (in the live app, one build runs per view per frame
    // with plenty of other work between).
    cudaEvent_t start, stop;
    CHECK_CU(cudaEventCreate(&start));
    CHECK_CU(cudaEventCreate(&stop));
    const int iterations = 100;
    float totalMilliseconds = 0;
    for (int i = 0; i < iterations; ++i) {
      CHECK_CU(cudaEventRecord(start, 0));
      once();
      CHECK_CU(cudaEventRecord(stop, 0));
      CHECK_CU(cudaEventSynchronize(stop));
      float ms = 0;
      CHECK_CU(cudaEventElapsedTime(&ms, start, stop));
      totalMilliseconds += ms;
    }
    printf("\nbuildAdaptiveStripMesh on 480x270: %.3f ms/frame (avg of %d)\n", totalMilliseconds / iterations, iterations);
    // Deep-queue timing: all iterations enqueued back-to-back, one sync at the end. This
    // hides the per-launch CPU overhead the synchronized loop exposes; the difference
    // between the two numbers is launch/sync overhead, not GPU work.
    CHECK_CU(cudaEventRecord(start, 0));
    for (int i = 0; i < iterations; ++i) once();
    CHECK_CU(cudaEventRecord(stop, 0));
    CHECK_CU(cudaEventSynchronize(stop));
    float deepQueueMilliseconds = 0;
    CHECK_CU(cudaEventElapsedTime(&deepQueueMilliseconds, start, stop));
    printf("buildAdaptiveStripMesh on 480x270: %.3f ms/frame deep-queued (avg of %d)\n", deepQueueMilliseconds / iterations, iterations);

    // Production-config timing: 15 degree merge angle (the bench default of 2 degrees
    // merges almost nothing, which maximizes emit traffic).
    pReal.mergeAngleRadians = 15.0f * (float) M_PI / 180.0f;
    for (int i = 0; i < 10; ++i) once();
    CHECK_CU(cuStreamSynchronize(0));
    totalMilliseconds = 0;
    for (int i = 0; i < iterations; ++i) {
      CHECK_CU(cudaEventRecord(start, 0));
      once();
      CHECK_CU(cudaEventRecord(stop, 0));
      CHECK_CU(cudaEventSynchronize(stop));
      float ms = 0;
      CHECK_CU(cudaEventElapsedTime(&ms, start, stop));
      totalMilliseconds += ms;
    }
    printf("buildAdaptiveStripMesh on 480x270 @ 15 deg: %.3f ms/frame (avg of %d)\n", totalMilliseconds / iterations, iterations);
    cuMemFree(d_vbo);
    cuMemFree(d_ibo);
    cuMemFree(d_args);
    nvtxRangePop();
  }

  if (g_failures == 0)
    printf("\nAll checks passed.\n");
  else
    printf("\n%d check(s) FAILED.\n", g_failures);
  return g_failures ? 1 : 0;
}
