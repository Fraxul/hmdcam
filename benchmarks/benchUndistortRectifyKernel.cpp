// Standalone micro-benchmark / NSight-Compute harness for launchUndistortRectifyKernel().
//
// The kernel generates an RG16 distortion (rectification) map: one CUDA thread per output map
// texel converts (x, y) -> source UV via a per-row inv(P*R) matrix, applies Brown-Conrady forward
// distortion, and surf2Dwrite's the packed ushort2 UV. It is the producer half of the pair whose
// consumer is remapArray().
//
// This harness reproduces the live inputs: the first camera's OpenCV intrinsics + distortion
// (copied from calibration.yml), a 1920x1080 input stream, and an RG16 output surface. It sweeps
// full/half/quarter map resolutions and reports CUDA-event min/avg/max per size.
//
// Per-row rolling-shutter matrices are identity (== plain cv::initUndistortRectifyMap): the kernel
// has no data-dependent control flow, so their contents do not affect timing.

#include "common/UndistortRectifyKernel.h"
#include "rhi/cuda/CudaUtil.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// ----- Fixed input stream geometry. -----
constexpr int kStreamWidth = 1920;
constexpr int kStreamHeight = 1080;

constexpr int kDefaultWarmupIterations = 5;
constexpr int kDefaultTimedIterations = 200;

// ----- First camera's intrinsics + Brown-Conrady distortion, copied verbatim from
// calibration.yml (cameras[0]). -----
// 3x3 row-major intrinsics: fx, 0, cx / 0, fy, cy / 0, 0, 1.
const float kCameraMatrix[9] = {
  910.29749298720685f,
  0.0f,
  938.97957817935833f,
  0.0f,
  910.29749298720685f,
  556.50857463621719f,
  0.0f,
  0.0f,
  1.0f,
};
// k1, k2, p1, p2, k3.
const float kDistCoeffs[5] = {
  -0.0059976693512363575f,
  -0.0074646424535317968f,
  -0.00030725353677605823f,
  -0.0010042741225002713f,
  0.00010085800732822053f,
};

struct Timing {
  double avgMs, minMs, maxMs;
};

} // namespace

int main(int argc, char** argv) {
  const int timedIterations = (argc > 1) ? atoi(argv[1]) : kDefaultTimedIterations;

  // ----- CUDA context. -----
  CUDA_CHECK(cuInit(0));
  CUdevice device = 0;
  CUDA_CHECK(cuDeviceGet(&device, 0));
  CUcontext context = nullptr;
  CUDA_CHECK(cuDevicePrimaryCtxRetain(&context, device));
  CUDA_CHECK(cuCtxSetCurrent(context));

  char deviceName[256] = {0};
  CUDA_CHECK(cuDeviceGetName(deviceName, sizeof(deviceName), device));
  printf("Device: %s\n", deviceName);
  printf("Input stream: %dx%d\n", kStreamWidth, kStreamHeight);
  printf("Timed iterations: %d (warm-up %d excluded)\n\n", timedIterations, kDefaultWarmupIterations);

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // ----- Map resolutions to benchmark. -----
  struct Size {
    const char* label;
    int width;
    int height;
  };
  const Size sizes[] = {
    {"full    1920x1080",     kStreamWidth,     kStreamHeight},
    { "half     960x540", kStreamWidth / 2, kStreamHeight / 2},
    { "quarter  480x270", kStreamWidth / 4, kStreamHeight / 4},
  };

  printf("%-20s %10s %10s %10s\n", "map size", "avg(ms)", "min(ms)", "max(ms)");
  printf("%-20s %10s %10s %10s\n", "--------", "-------", "-------", "-------");

  for (const Size& s : sizes) {
    // ----- Params: static per-camera calibration + the map/stream scale-bias. distortionMapTo-
    // StreamScale = stream/map, with a half-scale bias so a map texel lands in the center of its
    // corresponding stream-pixel neighborhood (matches CameraSystem: scale 2, bias 1 at half-res).
    UndistortRectifyParams hostParams;
    memset(&hostParams, 0, sizeof(hostParams));
    memcpy(hostParams.cameraMatrix, kCameraMatrix, sizeof(kCameraMatrix));
    memcpy(hostParams.distCoeffs, kDistCoeffs, sizeof(kDistCoeffs));
    hostParams.distortionMapWidth = static_cast<uint32_t>(s.width);
    hostParams.distortionMapHeight = static_cast<uint32_t>(s.height);
    hostParams.streamWidth = kStreamWidth;
    hostParams.streamHeight = kStreamHeight;
    const float scale = static_cast<float>(kStreamWidth) / static_cast<float>(s.width);
    hostParams.distortionMapToStreamScale[0] = scale;
    hostParams.distortionMapToStreamScale[1] = scale;
    hostParams.distortionMapToStreamBias[0] = 0.5f * scale;
    hostParams.distortionMapToStreamBias[1] = 0.5f * scale;

    UndistortRectifyParams* d_params = nullptr;
    CUDA_CHECK(cudaMalloc(&d_params, sizeof(UndistortRectifyParams)));
    CUDA_CHECK(cudaMemcpy(d_params, &hostParams, sizeof(UndistortRectifyParams), cudaMemcpyHostToDevice));

    // ----- Per-row iR: identity 3x3 for every output row (== plain initUndistortRectifyMap). -----
    std::vector<float> hostPerRow(static_cast<size_t>(s.height) * 9, 0.0f);
    for (int y = 0; y < s.height; ++y) {
      float* m = hostPerRow.data() + (static_cast<size_t>(y) * 9);
      m[0] = m[4] = m[8] = 1.0f;
    }
    float* d_perRow = nullptr;
    CUDA_CHECK(cudaMalloc(&d_perRow, hostPerRow.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_perRow, hostPerRow.data(), hostPerRow.size() * sizeof(float), cudaMemcpyHostToDevice));

    // ----- Output surface: RG16 CUDA array with surface load/store. -----
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(16, 16, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray_t outArray = nullptr;
    CUDA_CHECK(cudaMallocArray(&outArray, &channelDesc, s.width, s.height, cudaArraySurfaceLoadStore));

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = outArray;
    cudaSurfaceObject_t outSurface = 0;
    CUDA_CHECK(cudaCreateSurfaceObject(&outSurface, &resDesc));

    auto launch = [&]() {
      launchUndistortRectifyKernel(d_params, d_perRow, outSurface, s.width, s.height, stream);
    };

    // ----- Warm-up (excluded). -----
    for (int i = 0; i < kDefaultWarmupIterations; ++i)
      launch();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ----- Timed loop: one CUDA-event pair per iteration. -----
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
    const Timing t{totalMs / static_cast<double>(timedIterations), minMs, maxMs};
    printf("%-20s %10.4f %10.4f %10.4f\n", s.label, t.avgMs, t.minMs, t.maxMs);

    // ----- Per-size cleanup. -----
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaDestroySurfaceObject(outSurface));
    CUDA_CHECK(cudaFreeArray(outArray));
    CUDA_CHECK(cudaFree(d_params));
    CUDA_CHECK(cudaFree(d_perRow));
  }

  CUDA_CHECK(cudaStreamDestroy(stream));
  cuDevicePrimaryCtxRelease(device);
  return 0;
}
