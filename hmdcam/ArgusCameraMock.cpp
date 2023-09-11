#include "ArgusCameraMock.h"
#include "common/EnvVar.h"
#include "common/Timing.h"
#include "imgui.h"
#include "implot/implot.h"
#include "rhi/gl/GLCommon.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/CudaUtil.h"
#include "rhi/cuda/RHICVInterop.h"
#include <cudaEGL.h>
#include <opencv2/cvconfig.h>
#ifdef HAVE_VPI2
#include "common/VPIUtil.h"
#include <vpi/Image.h>
#endif // HAVE_VPI2

ArgusCameraMock::ArgusCameraMock(size_t sensorCount, unsigned int w, unsigned int h, double framerate) {
  m_streamWidth = w;
  m_streamHeight = h;
  m_targetCaptureIntervalNs = 1000000000.0 / framerate;

  m_frameMetadata.resize(sensorCount);
  for (size_t i = 0; i < m_frameMetadata.size(); ++i) {
    auto& md = m_frameMetadata[i];
    md.sensorTimestamp = 0;
    md.frameDurationNs = m_targetCaptureIntervalNs;
    md.sensorExposureTimeNs = m_targetCaptureIntervalNs;
    md.sensorSensitivityISO = 100;
    md.ispDigitalGain = 1.0f;
    md.sensorAnalogGain = 1.0f;
  }

  m_textures.resize(sensorCount);

  m_vpiImages.resize(sensorCount);
}

ArgusCameraMock::~ArgusCameraMock() {

}

bool ArgusCameraMock::readFrame() {
  uint64_t now = currentTimeNs();

  // Frame pacing
  uint64_t delta = now - m_previousFrameReadTime;
  if (delta < m_targetCaptureIntervalNs) {
    delayNs(m_targetCaptureIntervalNs - delta);
  }

  for (size_t i = 0; i < m_frameMetadata.size(); ++i) {
    m_frameMetadata[i].sensorTimestamp = now - m_targetCaptureIntervalNs;
  }
  for (size_t i = 0; i < m_textures.size(); ++i) {
    if (m_textures[i])
      continue;

    RHISurface::ptr srf = rhi()->newTexture2D(streamWidth(), streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    RHIRenderTarget::ptr rt = rhi()->compileRenderTarget(RHIRenderTargetDescriptor( { srf } ));

    switch (i) {
      case 0:
        rhi()->setClearColor(glm::vec4(1.0f, 0.0f, 0.0f, 1.0f)); break;
      case 1:
        rhi()->setClearColor(glm::vec4(0.0f, 1.0f, 0.0f, 1.0f)); break;
      case 2:
        rhi()->setClearColor(glm::vec4(0.0f, 0.0f, 1.0f, 1.0f)); break;
      case 3:
      default:
        rhi()->setClearColor(glm::vec4(1.0f, 0.0f, 1.0f, 1.0f)); break;
    };

    rhi()->beginRenderPass(rt, kLoadClear);
    rhi()->endRenderPass(rt);

    m_textures[i] = srf;

#ifdef HAVE_VPI2
    // Argus NVBuffers wrapped into VPIImage are VPI_IMAGE_FORMAT_NV12_ER
    VPI_CHECK(vpiImageCreate(streamWidth(), streamHeight(), VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_CUDA, &m_vpiImages[i]));
#endif
  }

  m_previousFrameReadTime = now;
  return true;
}

VPIImage ArgusCameraMock::vpiImage(size_t sensorIndex) const {
  return m_vpiImages[sensorIndex];
  return NULL;
}

cv::cuda::GpuMat ArgusCameraMock::gpuMatGreyscale(size_t sensorIdx) {
  assert(false && "Not implemented");
  return cv::cuda::GpuMat();
}


