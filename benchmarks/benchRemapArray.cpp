// Standalone micro-benchmark / NSight-Compute harness for remapArray() / remapArrayRGB().
//
// remapArray gathers source pixels through a rectification map: one CUDA thread per output
// pixel takes oversampleFactor^2 taps of the (float2) distortion map to find source UVs, then
// oversampleFactor^2 taps of the source texture, and averages. remapArrayRGB does the same but
// samples an NV12 luma+chroma pair and converts BT.601 full-range YUV->RGB.
//
// This harness reproduces the exact live inputs:
//   - a 1920x1080 8-bit luma source texture (+ a matching 960x540 NV12 chroma plane for RGB),
//     created with the same driver-API descriptor the camera provider uses
//     (linear filter, clamp, normalized coords -- see ArgusCameraMock.cpp);
//   - realistic RG16 distortion maps generated from the first camera's OpenCV calibration via
//     cv::initUndistortRectifyMap, packed exactly as CameraSystem::generateGPUDistortionMap.
//
// It then sweeps the (output size x oversampleFactor) matrix the live app exercises and reports
// CUDA-event min/avg/max for both the luma and RGB kernels.
//
// Map<->output pairing: the live distortion map is always half-res (streamWidth/2). We pair each
// output with a same-resolution map where one exists; the full-res (1920x1080) output has no
// full-res map, so it borrows the 960x540 half-res map (bilinear upsampling of the coord field).
// Map resolution only shifts texture-cache locality -- timing is dominated by outputPixels *
// oversampleFactor^2 -- so this pairing does not materially change the measured cost.

#include "common/remapArray.h"
#include "rhi/cuda/CudaUtil.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// ----- Fixed input geometry (see CLAUDE.md / calibration.yml). -----
constexpr int kStreamWidth = 1920;
constexpr int kStreamHeight = 1080;

// The unit under test is invoked many times; a handful of warm-up iterations are excluded from
// the reported statistics so we measure steady-state, not first-launch JIT/alloc cost.
constexpr int kDefaultWarmupIterations = 5;
constexpr int kDefaultTimedIterations = 200;

// ----- First camera's intrinsics + Brown-Conrady distortion, copied verbatim from
// calibration.yml (cameras[0]). Used to synthesize a realistic rectification map. -----
// 3x3 row-major intrinsics.
const double kCameraMatrix[9] = {
  910.29749298720685,
  0.0,
  938.97957817935833,
  0.0,
  910.29749298720685,
  556.50857463621719,
  0.0,
  0.0,
  1.0,
};
// k1, k2, p1, p2, k3.
const double kDistCoeffs[5] = {
  -0.0059976693512363575,
  -0.0074646424535317968,
  -0.00030725353677605823,
  -0.0010042741225002713,
  0.00010085800732822053,
};

// A distortion map texture plus the geometry it was generated at.
struct DistortionMap {
  int width = 0;
  int height = 0;
  CUarray array = nullptr;
  CUtexObject tex = 0;
};

// Build a realistic RG16 rectification map at the requested resolution and wrap it in a texture
// object matching CameraSystem's stereoDistortionCudaTexture (linear, clamp, normalized coords,
// read as normalized float). map1/map2 come from OpenCV so the sampled coordinate field -- and
// therefore the source-texture access pattern -- mirrors the live app.
DistortionMap createDistortionMap(int mapWidth, int mapHeight) {
  cv::Mat cameraMatrix(3, 3, CV_64F, const_cast<double*>(kCameraMatrix));
  cv::Mat distCoeffs(5, 1, CV_64F, const_cast<double*>(kDistCoeffs));

  // The rectified (new) camera renders the undistorted view at the map's resolution, so scale the
  // intrinsics by the map/stream ratio -- this is the identity-rectification analogue of the live
  // stereoProjection at half/quarter res. The map then holds source-pixel coords in stream space.
  const double scale = static_cast<double>(mapWidth) / static_cast<double>(kStreamWidth);
  cv::Mat newCameraMatrix = cameraMatrix.clone();
  newCameraMatrix.at<double>(0, 0) *= scale; // fx
  newCameraMatrix.at<double>(1, 1) *= scale; // fy
  newCameraMatrix.at<double>(0, 2) *= scale; // cx
  newCameraMatrix.at<double>(1, 2) *= scale; // cy

  cv::Mat map1, map2; // CV_32FC1 absolute source-pixel X and Y.
  cv::initUndistortRectifyMap(
    cameraMatrix, distCoeffs, cv::Mat::eye(3, 3, CV_64F), newCameraMatrix,
    cv::Size(mapWidth, mapHeight), CV_32FC1, map1, map2);

  // Pack to RG16 normalized UVs, matching CameraSystem::generateGPUDistortionMap (half-texel bias,
  // clamp to [0,1], scale to 65535).
  constexpr float texelBias = 0.5f;
  std::vector<uint16_t> packed(static_cast<size_t>(mapWidth) * mapHeight * 2);
  for (int y = 0; y < mapHeight; ++y) {
    uint16_t* rowP = packed.data() + (static_cast<size_t>(y) * mapWidth * 2);
    for (int x = 0; x < mapWidth; ++x) {
      const float u = (map1.at<float>(y, x) + texelBias) / static_cast<float>(kStreamWidth);
      const float v = (map2.at<float>(y, x) + texelBias) / static_cast<float>(kStreamHeight);
      *(rowP++) = static_cast<uint16_t>(std::min(std::max(u, 0.0f), 1.0f) * 65535.0f);
      *(rowP++) = static_cast<uint16_t>(std::min(std::max(v, 0.0f), 1.0f) * 65535.0f);
    }
  }

  DistortionMap out;
  out.width = mapWidth;
  out.height = mapHeight;

  // RG16-unorm CUDA array.
  CUDA_ARRAY_DESCRIPTOR arrayDesc;
  memset(&arrayDesc, 0, sizeof(arrayDesc));
  arrayDesc.Width = mapWidth;
  arrayDesc.Height = mapHeight;
  arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
  arrayDesc.NumChannels = 2;
  CUDA_CHECK(cuArrayCreate(&out.array, &arrayDesc));

  CUDA_MEMCPY2D copyDescriptor;
  memset(&copyDescriptor, 0, sizeof(copyDescriptor));
  copyDescriptor.srcMemoryType = CU_MEMORYTYPE_HOST;
  copyDescriptor.srcHost = packed.data();
  copyDescriptor.srcPitch = sizeof(uint16_t) * mapWidth * 2;
  copyDescriptor.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copyDescriptor.dstArray = out.array;
  copyDescriptor.WidthInBytes = sizeof(uint16_t) * mapWidth * 2;
  copyDescriptor.Height = mapHeight;
  CUDA_CHECK(cuMemcpy2D(&copyDescriptor));

  // Texture: linear filter, clamp, normalized coords. Default read of an integer array format is
  // normalized-float (== cudaReadModeNormalizedFloat), matching the live map texture.
  CUDA_RESOURCE_DESC resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
  resDesc.res.array.hArray = out.array;

  CUDA_TEXTURE_DESC texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
  texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
  texDesc.maxAnisotropy = 1;
  CUDA_CHECK(cuTexObjectCreate(&out.tex, &resDesc, &texDesc, /*resourceViewDescriptor=*/ nullptr));

  return out;
}

// Wrap a single/two-channel 8-bit pitched GpuMat in a texture object matching the camera
// provider's luma/chroma textures (linear filter, clamp, normalized coords).
CUtexObject createSourceTexture(const cv::cuda::GpuMat& mat, int numChannels) {
  CUDA_RESOURCE_DESC resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = CU_RESOURCE_TYPE_PITCH2D;
  resDesc.res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT8;
  resDesc.res.pitch2D.numChannels = numChannels;
  resDesc.res.pitch2D.width = mat.cols;
  resDesc.res.pitch2D.height = mat.rows;
  resDesc.res.pitch2D.pitchInBytes = mat.step;
  resDesc.res.pitch2D.devPtr = reinterpret_cast<CUdeviceptr>(mat.cudaPtr());

  CUDA_TEXTURE_DESC texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
  texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
  texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES;
  texDesc.maxAnisotropy = 1;

  CUtexObject tex = 0;
  CUDA_CHECK(cuTexObjectCreate(&tex, &resDesc, &texDesc, /*resourceViewDescriptor=*/ nullptr));
  return tex;
}

// One benchmark scenario: output size, oversampleFactor, and which map drives the gather.
struct Case {
  const char* label;
  int outWidth;
  int outHeight;
  unsigned int oversampleFactor;
  const DistortionMap* map;
};

struct Timing {
  double avgMs, minMs, maxMs;
};

// Time a kernel launch closure with one CUDA-event pair per iteration (warm-up excluded).
template <typename Fn>
Timing timeKernel(Fn&& launch, cudaStream_t stream, int timedIterations) {
  for (int i = 0; i < kDefaultWarmupIterations; ++i)
    launch();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
  CUDA_CHECK(cudaEventCreate(&startEvent));
  CUDA_CHECK(cudaEventCreate(&stopEvent));

  double totalMs = 0.0, minMs = 1e30, maxMs = 0.0;
  for (int i = 0; i < timedIterations; ++i) {
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    launch();
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, startEvent, stopEvent));
    totalMs += ms;
    if (ms < minMs) minMs = ms;
    if (ms > maxMs) maxMs = ms;
  }

  CUDA_CHECK(cudaEventDestroy(startEvent));
  CUDA_CHECK(cudaEventDestroy(stopEvent));
  return Timing{totalMs / static_cast<double>(timedIterations), minMs, maxMs};
}

} // namespace

int main(int argc, char** argv) {
  const int timedIterations = (argc > 1) ? atoi(argv[1]) : kDefaultTimedIterations;

  // ----- CUDA context: retain the device primary context so driver-API allocations share a
  // context with OpenCV's runtime-API GpuMat allocations. -----
  CUDA_CHECK(cuInit(0));
  CUdevice device = 0;
  CUDA_CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CUDA_CHECK(cuCtxSetCurrent(context));

  char deviceName[256] = {0};
  CUDA_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), device));
  printf("Device: %s\n", deviceName);
  printf("Source: %dx%d 8-bit luma (+ %dx%d NV12 chroma)\n",
    kStreamWidth, kStreamHeight, kStreamWidth / 2, kStreamHeight / 2);
  printf("Timed iterations: %d (warm-up %d excluded)\n\n", timedIterations, kDefaultWarmupIterations);

  // ----- Synthetic source planes. A smooth gradient + a little structure exercises the linear
  // filter realistically; exact contents don't affect timing (no data-dependent branching). -----
  cv::Mat hostLuma(kStreamHeight, kStreamWidth, CV_8UC1);
  for (int y = 0; y < kStreamHeight; ++y) {
    uint8_t* row = hostLuma.ptr<uint8_t>(y);
    for (int x = 0; x < kStreamWidth; ++x)
      row[x] = static_cast<uint8_t>((x ^ y) + (x >> 2));
  }
  cv::cuda::GpuMat lumaGpu;
  lumaGpu.upload(hostLuma);

  // NV12 chroma: half-res, two interleaved channels (Cb, Cr).
  cv::Mat hostChroma(kStreamHeight / 2, kStreamWidth / 2, CV_8UC2);
  for (int y = 0; y < hostChroma.rows; ++y) {
    cv::Vec2b* row = hostChroma.ptr<cv::Vec2b>(y);
    for (int x = 0; x < hostChroma.cols; ++x)
      row[x] = cv::Vec2b(static_cast<uint8_t>(x + y), static_cast<uint8_t>(x - y));
  }
  cv::cuda::GpuMat chromaGpu;
  chromaGpu.upload(hostChroma);

  const CUtexObject lumaTex = createSourceTexture(lumaGpu, /*numChannels=*/ 1);
  const CUtexObject chromaTex = createSourceTexture(chromaGpu, /*numChannels=*/ 2);

  // ----- Distortion maps: half-res and quarter-res. -----
  DistortionMap halfMap = createDistortionMap(kStreamWidth / 2, kStreamHeight / 2); // 960x540
  DistortionMap quarterMap = createDistortionMap(kStreamWidth / 4, kStreamHeight / 4); // 480x270
  printf("Distortion maps: half=%dx%d, quarter=%dx%d\n\n",
    halfMap.width, halfMap.height, quarterMap.width, quarterMap.height);

  // ----- Scenario matrix (see file header for the map<->output pairing rationale). -----
  const Case cases[] = {
    {   "full  1920x1080 os=1 (half map)",     kStreamWidth,     kStreamHeight, 1,    &halfMap},
    {   "half   960x540  os=1 (half map)", kStreamWidth / 2, kStreamHeight / 2, 1,    &halfMap},
    {   "half   960x540  os=2 (half map)", kStreamWidth / 2, kStreamHeight / 2, 2,    &halfMap},
    {"quarter480x270  os=1 (quarter map)", kStreamWidth / 4, kStreamHeight / 4, 1, &quarterMap},
    {"quarter480x270  os=2 (quarter map)", kStreamWidth / 4, kStreamHeight / 4, 2, &quarterMap},
    {"quarter480x270  os=4 (quarter map)", kStreamWidth / 4, kStreamHeight / 4, 4, &quarterMap},
  };

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  printf("%-38s %-6s %10s %10s %10s\n", "scenario", "kernel", "avg(ms)", "min(ms)", "max(ms)");
  printf("%-38s %-6s %10s %10s %10s\n", "--------", "------", "-------", "-------", "-------");

  for (const Case& c : cases) {
    // remapArray (luma -> CV_8U).
    cv::cuda::GpuMat dstLuma(c.outHeight, c.outWidth, CV_8UC1);
    const Timing tLuma = timeKernel([&]() {
      remapArray(lumaTex, c.map->tex, dstLuma, reinterpret_cast<CUstream>(stream), c.oversampleFactor);
    },
      stream, timedIterations);
    printf("%-38s %-6s %10.4f %10.4f %10.4f\n", c.label, "luma", tLuma.avgMs, tLuma.minMs, tLuma.maxMs);

    // remapArrayRGB (luma + chroma -> CV_8UC3).
    cv::cuda::GpuMat dstRGB(c.outHeight, c.outWidth, CV_8UC3);
    const Timing tRGB = timeKernel([&]() {
      remapArrayRGB(lumaTex, chromaTex, c.map->tex, dstRGB, reinterpret_cast<CUstream>(stream), c.oversampleFactor);
    },
      stream, timedIterations);
    printf("%-38s %-6s %10.4f %10.4f %10.4f\n", "", "rgb", tRGB.avgMs, tRGB.minMs, tRGB.maxMs);
  }

  // ----- Cleanup. -----
  CUDA_CHECK(cudaStreamDestroy(stream));
  cuTexObjectDestroy(lumaTex);
  cuTexObjectDestroy(chromaTex);
  cuTexObjectDestroy(halfMap.tex);
  cuTexObjectDestroy(quarterMap.tex);
  cuArrayDestroy(halfMap.array);
  cuArrayDestroy(quarterMap.array);
  cuDevicePrimaryCtxRelease(device);
  return 0;
}
