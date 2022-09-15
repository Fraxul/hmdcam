#pragma once
#ifdef HAVE_OPENCV_CUDA
#include "IDebugOverlay.h"
#include "rhi/RHISurface.h"
#include <opencv2/core/cuda.hpp>
#include <nppcore.h>

class ICameraProvider;


class FocusAssistDebugOverlay : public IDebugOverlay {
public:
  FocusAssistDebugOverlay(ICameraProvider*);
  virtual ~FocusAssistDebugOverlay();

  virtual DebugOverlayType overlayType() const { return kDebugOverlayFocusAssist; }

  virtual void update();
  virtual void renderIMGUI();
  virtual RHISurface::ptr overlaySurfaceForCamera(size_t cameraIdx);

  ICameraProvider* cameraProvider() const { return m_cameraProvider; }

protected:
  ICameraProvider* m_cameraProvider;

  struct PerCameraData;
  PerCameraData* m_perCameraData;

  cv::cuda::Stream m_cudaStream;
};

#endif // HAVE_OPENCV_CUDA
