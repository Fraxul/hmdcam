#include "common/UndistortRectifyKernel.h"
#include <cuda_runtime.h>

namespace {

inline int divUp(int n, int d) { return (n + d - 1) / d; }

__global__ void undistortRectifyKernel(
  const UndistortRectifyParams* __restrict__ params,
  const float* __restrict__ invRectifiedProjectionPerRow,
  cudaSurfaceObject_t outMap) {

  const int32_t width = params->width;
  const int32_t height = params->height;

  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const float fx = params->cameraMatrix[0];
  const float fy = params->cameraMatrix[4];
  const float cx = params->cameraMatrix[2];
  const float cy = params->cameraMatrix[5];

  const float k1 = params->distCoeffs[0];
  const float k2 = params->distCoeffs[1];
  const float p1 = params->distCoeffs[2];
  const float p2 = params->distCoeffs[3];
  const float k3 = params->distCoeffs[4];

  // Per-row iR: all threads in a warp share y (block is 32x8, warp packs along
  // threadIdx.x first), so the 9 loads broadcast through L1 as a single read.
  const float* iR = invRectifiedProjectionPerRow + (y * 9);

  // (u, v, 1) -> ray in source camera's normalized frame at this row's
  // exposure time, via inv(P * R) pre-folded with the per-row rolling-shutter
  // correction.
  const float u = static_cast<float>(x);
  const float v = static_cast<float>(y);
  const float xR = u * iR[0] + v * iR[1] + iR[2];
  const float yR = u * iR[3] + v * iR[4] + iR[5];
  const float wR = u * iR[6] + v * iR[7] + iR[8];

  // Using the __fdividef fast reciprocal/divide intrinsic here and in the coordinate
  // conversion below saves about 9% runtime, and the reduced precision is not
  // noticeable in the output.
  const float invW = __fdividef(1.0f, wR);
  const float xn = xR * invW;
  const float yn = yR * invW;

  // Brown-Conrady forward distortion.
  const float r2 = xn * xn + yn * yn;
  const float radial = 1.0f + r2 * (k1 + r2 * (k2 + r2 * k3));
  const float xy2 = 2.0f * xn * yn;
  const float xd = xn * radial + p1 * xy2 + p2 * (r2 + 2.0f * xn * xn);
  const float yd = yn * radial + p2 * xy2 + p1 * (r2 + 2.0f * yn * yn);

  const float uSrc = fx * xd + cx;
  const float vSrc = fy * yd + cy;

  const float fwidth = static_cast<float>(width);
  const float fheight = static_cast<float>(height);
  const float nx = __saturatef(__fdividef(uSrc + params->texelBiasX, fwidth));
  const float ny = __saturatef(__fdividef(vSrc + params->texelBiasY, fheight));

  // Round-toward-zero (== floor for non-negative) to match the CPU baseline's
  // static_cast<uint16_t>() truncation.
  ushort2 packed;
  packed.x = static_cast<unsigned short>(__float2uint_rz(nx * 65535.0f));
  packed.y = static_cast<unsigned short>(__float2uint_rz(ny * 65535.0f));

  // surf2Dwrite x-offset is in BYTES, not elements.
  surf2Dwrite(packed, outMap, x * static_cast<int>(sizeof(ushort2)), y);
}

} // anonymous namespace

extern "C" void launchUndistortRectifyKernel(
  const UndistortRectifyParams* paramsDevicePtr,
  const float* invRectifiedProjectionPerRowDevicePtr,
  cudaSurfaceObject_t outDistortionMap,
  int width, int height,
  cudaStream_t stream) {
  const dim3 block(32, 8);
  const dim3 grid(divUp(width, block.x), divUp(height, block.y));
  undistortRectifyKernel<<<grid, block, 0, stream>>>(
    paramsDevicePtr,
    invRectifiedProjectionPerRowDevicePtr,
    outDistortionMap);
}
