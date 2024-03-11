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
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#ifdef HAVE_VPI2
#include "common/VPIUtil.h"
#include <vpi/Image.h>
#include <vpi/OpenCVInterop.hpp>
#include <vpi/Stream.h>
#include <vpi/algo/ConvertImageFormat.h>
#endif // HAVE_VPI2

bool skipFramePacing = false;

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

  m_streamData.resize(sensorCount);

  readEnvironmentVariable("MOCK_SKIP_FRAME_PACING", skipFramePacing);
}

ArgusCameraMock::~ArgusCameraMock() {

}

bool ArgusCameraMock::readFrame() {
  uint64_t now = currentTimeNs();

  // Frame pacing
  if (!skipFramePacing) {
    uint64_t delta = now - m_previousFrameReadTime;
    if (delta < m_targetCaptureIntervalNs) {
      delayNs(m_targetCaptureIntervalNs - delta);
    }
  }

  for (size_t i = 0; i < m_frameMetadata.size(); ++i) {
    m_frameMetadata[i].sensorTimestamp = now - m_targetCaptureIntervalNs;
  }

#ifdef HAVE_VPI2
  VPIStream vpiStream = nullptr;

#endif
  for (size_t cameraIdx = 0; cameraIdx < m_streamData.size(); ++cameraIdx) {
    Stream& stream = m_streamData[cameraIdx];

    if (stream.rgbTexture)
      continue; // already initialized

#ifdef HAVE_VPI2
    if (!vpiStream) {
      VPI_CHECK(vpiStreamCreate(VPI_BACKEND_CPU | VPI_BACKEND_VIC, &vpiStream));
    }
#endif

    RHISurface::ptr srf = rhi()->newTexture2D(streamWidth(), streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    stream.rgbTexture = srf;

#ifdef HAVE_VPI2
    // Argus NVBuffers wrapped into VPIImage are VPI_IMAGE_FORMAT_NV12_ER
    VPI_CHECK(vpiImageCreate(streamWidth(), streamHeight(), VPI_IMAGE_FORMAT_NV12_ER, VPI_BACKEND_VIC | VPI_BACKEND_CUDA, &stream.vpiImage));
#endif

    bool haveImageData = false;
    char filename[32];
    try {
      sprintf(filename, "camera%zu.png", cameraIdx);
      cv::Mat m = cv::imread(filename, cv::IMREAD_COLOR);
      if (m.cols == streamWidth() && m.rows == streamHeight()) {
        cv::Mat rgbaMat, lumaMat;
        cv::cvtColor(m, rgbaMat, cv::COLOR_BGRA2RGBA);
        cv::cvtColor(m, lumaMat, cv::COLOR_BGRA2GRAY);
        rhi()->loadTextureData(srf, kVertexElementTypeUByte4N, rgbaMat.data);

        stream.lumaGpuMat.create(lumaMat.rows, lumaMat.cols, lumaMat.type());
        stream.lumaGpuMat.upload(lumaMat);

        CUDA_RESOURCE_DESC lumaResourceDescriptor;
        memset(&lumaResourceDescriptor, 0, sizeof(lumaResourceDescriptor));
        lumaResourceDescriptor.resType = CU_RESOURCE_TYPE_PITCH2D;
        lumaResourceDescriptor.res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT8;
        lumaResourceDescriptor.res.pitch2D.numChannels = 1;
        lumaResourceDescriptor.res.pitch2D.width = stream.lumaGpuMat.cols;
        lumaResourceDescriptor.res.pitch2D.height = stream.lumaGpuMat.rows;
        lumaResourceDescriptor.res.pitch2D.pitchInBytes = stream.lumaGpuMat.step;
        lumaResourceDescriptor.res.pitch2D.devPtr = (CUdeviceptr) stream.lumaGpuMat.cudaPtr();

        CUDA_TEXTURE_DESC texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
        texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
        // texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES; // optional
        texDesc.maxAnisotropy = 1;

        CUDA_CHECK(cuTexObjectCreate(&stream.cudaLumaTexObject, &lumaResourceDescriptor, &texDesc, /*resourceViewDescriptor=*/ nullptr));

#ifdef HAVE_VPI2
        VPIImage vpiRGBAImage;
        VPI_CHECK(vpiImageCreateWrapperOpenCVMat(rgbaMat, VPI_IMAGE_FORMAT_RGBA8, VPI_BACKEND_CPU | VPI_BACKEND_VIC, &vpiRGBAImage));

        VPIConvertImageFormatParams convertParams;
        vpiInitConvertImageFormatParams(&convertParams);
        VPI_CHECK(vpiSubmitConvertImageFormat(vpiStream, VPI_BACKEND_VIC, vpiRGBAImage, stream.vpiImage, &convertParams));
        vpiStreamSync(vpiStream);
        vpiImageDestroy(vpiRGBAImage);
#endif

        haveImageData = true;
      } else {
        printf("ArgusCameraMock: loaded %s but the dimensions didn't match (expected %ux%u, got %ux%u)\n",
          filename, streamWidth(), streamHeight(), m.cols, m.rows);
      }
    } catch (const std::exception& ex) {
      printf("ArgusCameraMock: Couldn't open %s: %s\n", filename, ex.what());
    }

    if (!haveImageData) {
      // If we didn't load mock data for this camera, clear the surface to a solid color.
      RHIRenderTarget::ptr rt = rhi()->compileRenderTarget(RHIRenderTargetDescriptor( { srf } ));

      switch (cameraIdx) {
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
    }

  }

#ifdef HAVE_VPI2
    if (vpiStream) {
      vpiStreamDestroy(vpiStream);
    }
#endif

  m_previousFrameReadTime = now;
  return true;
}

VPIImage ArgusCameraMock::vpiImage(size_t sensorIndex) const {
  return m_streamData[sensorIndex].vpiImage;
}

