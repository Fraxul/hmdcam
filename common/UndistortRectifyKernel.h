#pragma once
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

// Static per-camera params for the undistort/rectify map generator. Mirrors the
// fixed inputs of cv::initUndistortRectifyMap. Passed by value to land in the
// constant bank, since access is globally uniform.
struct UndistortRectifyParams {
  // Source camera intrinsics (the camera that captured the input image), 3x3
  // row-major. The kernel reads only fx=[0,0], fy=[1,1], cx=[0,2], cy=[1,2];
  // the rest is unused but kept full-matrix to mirror the OpenCV convention.
  float cameraMatrix[9];

  // Brown-Conrady distortion: k1, k2, p1, p2, k3.
  float distCoeffs[5];

  // Output map dimensions in pixels.
  uint32_t distortionMapWidth, distortionMapHeight;

  // Input camera stream dimensions in pixels.
  uint32_t streamWidth, streamHeight;

  // Scale-bias to convert from distortion map pixel space to input camera pixel space.
  // streamPixel = (distortionMapPixel * distortionMapScale) + distortionMapBias
  float distortionMapToStreamScale[2];
  float distortionMapToStreamBias[2];
};

// Per-output-row 3x3 matrices for rolling-shutter correction. Layout: tightly
// packed row-major matrices indexed as [y * 9 + r * 3 + c], where y is the
// output row, r/c index the matrix element.
//
// Each entry is R_y * iR, where:
//   iR  = inv(newProjection_3x3 * rectification) -- the static, calibration-
//         derived inverse (what plain cv::initUndistortRectifyMap uses for the
//         entire frame).
//   R_y = rolling-shutter rotation taking the camera-frame orientation at the
//         reference time to its orientation at row y's exposure time. R_y == I
//         disables the rolling-shutter correction for that row, and a buffer
//         of all-identity-times-iR reproduces plain initUndistortRectifyMap.
//
// SYNCHRONIZATION ASSUMPTION: the caller must not rewrite this buffer while a
// kernel that reads it is in flight. The production plan was double-buffered
// pinned allocations indexed by frame parity, but we're skipping that for now:
// the camera capture pipeline gives us comfortable per-frame slack, and if
// rolling-shutter correction starts wobbling the system has bigger problems
// than this.

#ifdef __cplusplus
extern "C"
#endif
  void
  launchUndistortRectifyKernel(
    const UndistortRectifyParams& params,
    const float* invRectifiedProjectionPerRowDevicePtr,
    cudaSurfaceObject_t outDistortionMap,
    int distortionMapWidth, int distortionMapHeight,
    cudaStream_t stream);
