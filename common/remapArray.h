#pragma once
#include <cuda.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cuda.hpp>

// Remaps src into dst using the rectification map in undistortRectifyMap.
// oversampleFactor controls how many sub-samples are taken and averaged per pixel of dst;
// useful when there is a large difference in scale between src/input and dest/output.
//
// src must use linear filtering and normalized texture coordinates.
// dst must already exist at the desired output size and must be CV_8U.
// undistortRectifyMap must use linear filtering and normalized texture coordinates.
// oversampleFactor must be between 1-4 (inclusive)
void remapArray(CUtexObject src, CUtexObject undistortRectifyMap, cv::cuda::GpuMat& dst, CUstream stream, unsigned int oversampleFactor = 1);

// As remapArray(), but samples a luma + chroma (NV12) source pair and writes a packed RGB result,
// converting YUV->RGB with the BT.601 full-range matrix (matching the camera provider's
// VkSamplerYcbcrConversion: kYcbcrModelYcbcr601 / kYcbcrRangeFull).
//
// lumaTex / chromaTex must use linear filtering and normalized texture coordinates (as provided by
// ICameraProvider::cudaLumaTexObject() / cudaChromaTexObject()). The chroma texture is sampled as
// float2 (Cb, Cr); its half-resolution subsampling is handled by the texture unit.
// dst must already exist at the desired output size and must be CV_8UC3 (R, G, B byte order).
// undistortRectifyMap must use linear filtering and normalized texture coordinates.
// oversampleFactor must be between 1-4 (inclusive); Y and chroma are averaged across the subsamples.
void remapArrayRGB(CUtexObject lumaTex, CUtexObject chromaTex, CUtexObject undistortRectifyMap, cv::cuda::GpuMat& dst, CUstream stream, unsigned int oversampleFactor = 1);
