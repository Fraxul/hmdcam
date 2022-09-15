#ifdef HAVE_OPENCV_CUDA

#include "FocusAssistDebugOverlay.h"
#include "common/ICameraProvider.h"
#include "common/VPIUtil.h"
#include "imgui.h"
#include "rhi/RHI.h"
#include "rhi/cuda/RHICVInterop.h"
#include "rhi/cuda/CudaUtil.h"
#include <opencv2/cudafilters.hpp>
#include <array>
#include <nppi.h>

// Separable Sobel filter coefficients
static const std::array<float, 7> kSobelFilterRow = {-1, -5, -6, 0, +6, +5, +1};
static const std::array<float, 7> kSobelFilterCol = {1/64.f, 6/64.f, 15/64.f, 20/64.f, 15/64.f, 6/64.f, 1/64.f};

struct FocusAssistDebugOverlay::PerCameraData {
  cv::Ptr<cv::cuda::Filter> sobel;
  cv::cuda::GpuMat filteredMat;
  RHISurface::ptr overlaySurface;

};

FocusAssistDebugOverlay::FocusAssistDebugOverlay(ICameraProvider* c) : m_cameraProvider(c) {
  m_perCameraData = new PerCameraData[cameraProvider()->streamCount()];
}

FocusAssistDebugOverlay::~FocusAssistDebugOverlay() {
  delete[] m_perCameraData;
}


void FocusAssistDebugOverlay::update() {
  for (size_t cameraIdx = 0; cameraIdx < cameraProvider()->streamCount(); ++cameraIdx) {
    PerCameraData& d = m_perCameraData[cameraIdx];
    if (!d.overlaySurface) {
      d.overlaySurface = rhi()->newTexture2D(cameraProvider()->streamWidth(), cameraProvider()->streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));
    }

    cv::cuda::GpuMat srcMat = cameraProvider()->gpuMatGreyscale(cameraIdx);

    if (!d.sobel)
      d.sobel = cv::cuda::createSeparableLinearFilter(srcMat.type(), -1, kSobelFilterRow, kSobelFilterCol);

    d.sobel->apply(srcMat, d.filteredMat, m_cudaStream);
  }

  if (nppGetStream() != ((CUstream) m_cudaStream.cudaPtr())) {
    nppSetStream((CUstream) m_cudaStream.cudaPtr());
  }

  for (size_t cameraIdx = 0; cameraIdx < cameraProvider()->streamCount(); ++cameraIdx) {
    PerCameraData& d = m_perCameraData[cameraIdx];
    // 1-channel 8-bit unsigned char in-place threshold, "less than" comparison
    NppiSize sz;
    sz.width  = d.filteredMat.cols;
    sz.height = d.filteredMat.rows;
    nppiThreshold_LTVal_8u_C1IR(d.filteredMat.ptr<Npp8u>(), static_cast<int>(d.filteredMat.step), sz, /*comparisonValue=*/ 255, /*replacementValue=*/ 0);
  }

  for (size_t cameraIdx = 0; cameraIdx < cameraProvider()->streamCount(); ++cameraIdx) {
    PerCameraData& d = m_perCameraData[cameraIdx];
    RHICUDA::copyGpuMatToSurface(d.filteredMat, d.overlaySurface, m_cudaStream);
  }
}


void FocusAssistDebugOverlay::renderIMGUI() {

}

RHISurface::ptr FocusAssistDebugOverlay::overlaySurfaceForCamera(size_t cameraIdx) {
  return m_perCameraData[cameraIdx].overlaySurface;
}

#endif // HAVE_OPENCV_CUDA
