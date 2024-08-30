#include "common/DepthMapGeneratorMock.h"
#include "imgui.h"
#include "implot/implot.h"
#include "common/CameraSystem.h"
#include "common/ICameraProvider.h"
#include "common/Timing.h"
#include "common/glmCvInterop.h"
#include "common/remapArray.h"
#include "rhi/RHI.h"
#include "rhi/RHIResources.h"
#include "rhi/cuda/RHICVInterop.h"
#include "rhi/gl/GLCommon.h"
#include <opencv2/cvconfig.h>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <cuda.h>
#include <npp.h>
#include <epoxy/gl.h> // epoxy_is_desktop_gl

#include <stdlib.h>
#include <libgen.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <limits.h>
#include <string>

#define PER_EYE for (size_t eyeIdx = 0; eyeIdx < 2; ++eyeIdx)

DepthMapGeneratorMock::DepthMapGeneratorMock() : DepthMapGenerator(kDepthBackendMock) {

  // running at quarter res, approx
  m_algoDownsampleX = 4;
  m_algoDownsampleY = 4;
  m_maxDisparity = 128;
  m_disparityPrescale = (1.0f / 16.0f); // emulating 4 subpixel bits
  m_useFP16Disparity = false;
}

DepthMapGeneratorMock::~DepthMapGeneratorMock() {

}

void DepthMapGeneratorMock::internalLoadSettings(cv::FileStorage& fs) {

}

void DepthMapGeneratorMock::internalSaveSettings(cv::FileStorage& fs) {

}

void DepthMapGeneratorMock::internalUpdateViewData() {
  for (size_t viewIdx = 0; viewIdx < m_viewData.size(); ++viewIdx) {
    // CameraSystem::View& v = m_cameraSystem->viewAtIndex(viewIdx);
    auto vd = viewDataAtIndex(viewIdx);

    if (!vd->m_isStereoView)
      continue;

    vd->updateDisparityTexture(internalWidth(), internalHeight(), kSurfaceFormat_R16i);

    vd->fakeDisparity = cv::Mat(internalHeight(), internalWidth(), CV_16U);

    // TODO: Optionally load a disparity map from disk
    // Otherwise, fill with a constant value
    for (size_t y = 0; y < vd->fakeDisparity.rows; ++y) {
      for (size_t x = 0; x < vd->fakeDisparity.cols; ++x) {

        vd->fakeDisparity.ptr<uint16_t>(y)[x] = 64;
      }
    }
  }
}


void DepthMapGeneratorMock::internalProcessFrame() {

  // Fill render surfaces with fake data
  for (size_t viewIdx = 0; viewIdx < m_cameraSystem->views(); ++viewIdx) {
    auto vd = viewDataAtIndex(viewIdx);
    if (!vd->m_isStereoView)
      continue;

    if (m_debugDisparityCPUAccessEnabled)
      vd->ensureDebugCPUAccessEnabled(/*bytesPerPixel=*/ 2);

    rhi()->loadTextureData(vd->m_disparityTexture, kVertexElementTypeShort1, vd->fakeDisparity.ptr<uint16_t>(0));

    if (m_debugDisparityCPUAccessEnabled)
      memcpy(vd->m_debugCPUDisparity, vd->fakeDisparity.ptr<uint16_t>(0), sizeof(uint16_t) * vd->fakeDisparity.cols * vd->fakeDisparity.rows);
#if 0
    if (m_populateDebugTextures) {
      if (!vd->m_leftGray)
        vd->m_leftGray = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      if (!vd->m_rightGray)
        vd->m_rightGray = rhi()->newTexture2D(internalWidth(), internalHeight(), RHISurfaceDescriptor(kSurfaceFormat_R8));

      RHICUDA::copyGpuMatToSurface(vd->resizedLeft_gpu, vd->m_leftGray, m_globalStream);
      RHICUDA::copyGpuMatToSurface(vd->resizedRight_gpu, vd->m_rightGray, m_globalStream);
    }
#endif
  }

  internalGenerateDisparityMips();
}

void DepthMapGeneratorMock::internalRenderIMGUI() {
}

void DepthMapGeneratorMock::internalRenderIMGUIPerformanceGraphs() {
}

