#pragma once
#include <cstdint>
#include <cuda.h>
#include <opencv2/core.hpp>
#include "nvscibuf.h"

// Small NvSci/CUDA surface helpers for the OFA flow source, ported from the nvmedia-iep
// OFA test utilities and adapted to use RHICUDA::cudaDevice (the in-tree CUDA context) and
// the in-tree NvSciCudaInterop/NvSciUtil. Tegra-only.

namespace CameraImuCalib {

// Populate an NvSciBufAttrList for a single-plane image surface of the given size/format.
// The caller must already have run NvMediaIOFAFillNvSciBufAttrList on the list. setColorStd
// emits PlaneColorStd (REC709_ER) for the Y8 inputs and is omitted for signed/cost surfaces.
void populateImageBufAttrList(NvSciBufAttrList& attrList, uint32_t width, uint32_t height,
  NvSciBufAttrValColorFmt colorFmt, bool setColorStd);

// Host (cv::Mat) <-> CUDA array (mapped NvSciBuf surface) 2D DMA. Mat type must match the
// surface format (CV_8UC1 for Y8, CV_16SC2 for Signed_R16G16, CV_8UC1 for A8).
void copyCvMatToSurface(const cv::Mat& mat, CUarray arr, CUstream stream);
void copySurfaceToCvMat(CUarray arr, cv::Mat& mat, CUstream stream);

} // namespace CameraImuCalib
