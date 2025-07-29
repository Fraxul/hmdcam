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

  for (size_t cameraIdx = 0; cameraIdx < m_streamData.size(); ++cameraIdx) {
    Stream& stream = m_streamData[cameraIdx];

    if (stream.rgbTexture)
      continue; // already initialized

    RHISurface::ptr srf = rhi()->newTexture2D(streamWidth(), streamHeight(), RHISurfaceDescriptor(kSurfaceFormat_RGBA8));
    stream.rgbTexture = srf;

    bool haveImageData = false;
    char filename[32];
    cv::Mat m;
    try {
      sprintf(filename, "camera%zu.png", cameraIdx);
      m = cv::imread(filename, cv::IMREAD_COLOR); // input will be in BGRA format
      if (m.cols == streamWidth() && m.rows == streamHeight()) {
        cv::Mat rgbaMat;
        cv::cvtColor(m, rgbaMat, cv::COLOR_BGRA2RGBA);
        rhi()->loadTextureData(srf, kVertexElementTypeUByte4N, rgbaMat.data);
        haveImageData = true;
      } else {
        printf("ArgusCameraMock: loaded %s but the dimensions didn't match (expected %ux%u, got %ux%u)\n",
          filename, streamWidth(), streamHeight(), m.cols, m.rows);
      }
    } catch (const std::exception& ex) {
      printf("ArgusCameraMock: Couldn't open %s: %s\n", filename, ex.what());
    }

    if (haveImageData) {
      // CUDA resource setup. This will fail on a (probably WSL2) machine with no CUDA devices,
      // but we may be able to continue anyway.
      try {
        cv::Mat lumaMat, chromaMat;
        cv::Mat yv12Mat;
        {
          // Convert from input BGRA format to YV12
          // The resultant image should be CV_8UC1, same width, and 1.5x the height of the input.
          // The luma plane is packed first, followed by the half-res chroma

          cv::cvtColor(m, yv12Mat, cv::COLOR_BGRA2YUV_YV12);
          assert(yv12Mat.rows == (m.rows + (m.rows / 2)));
          assert(yv12Mat.cols == m.cols);
          assert(yv12Mat.type() == CV_8UC1);

          lumaMat = yv12Mat.rowRange(0, m.rows);
          chromaMat = yv12Mat.rowRange(m.rows, yv12Mat.rows).reshape(/*channels=*/ 2);
        }
        printf("Stream [%zu]:   Luma: %u x %u, %u channels, %zu bytes/channel\n", cameraIdx, lumaMat.cols, lumaMat.rows, lumaMat.channels(), lumaMat.elemSize1());
        printf("Stream [%zu]: Chroma: %u x %u, %u channels, %zu bytes/channel\n", cameraIdx, chromaMat.cols, chromaMat.rows, chromaMat.channels(), chromaMat.elemSize1());

        stream.lumaGpuMat.create(lumaMat.rows, lumaMat.cols, lumaMat.type());
        stream.lumaGpuMat.upload(lumaMat);

        stream.chromaGpuMat.create(chromaMat.rows, chromaMat.cols, chromaMat.type());
        stream.chromaGpuMat.upload(chromaMat);

        {
          // Luma
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
        }

        {
          // Chroma -- half-res 2-channel uint8
          CUDA_RESOURCE_DESC chromaResourceDescriptor;
          memset(&chromaResourceDescriptor, 0, sizeof(chromaResourceDescriptor));
          chromaResourceDescriptor.resType = CU_RESOURCE_TYPE_PITCH2D;
          chromaResourceDescriptor.res.pitch2D.format = CU_AD_FORMAT_UNSIGNED_INT8;
          chromaResourceDescriptor.res.pitch2D.numChannels = 2;
          chromaResourceDescriptor.res.pitch2D.width = stream.chromaGpuMat.cols;
          chromaResourceDescriptor.res.pitch2D.height = stream.chromaGpuMat.rows;
          chromaResourceDescriptor.res.pitch2D.pitchInBytes = stream.chromaGpuMat.step;
          chromaResourceDescriptor.res.pitch2D.devPtr = (CUdeviceptr) stream.chromaGpuMat.cudaPtr();

          CUDA_TEXTURE_DESC texDesc;
          memset(&texDesc, 0, sizeof(texDesc));
          texDesc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
          texDesc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
          texDesc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
          texDesc.filterMode = CU_TR_FILTER_MODE_LINEAR;
          // texDesc.flags = CU_TRSF_NORMALIZED_COORDINATES; // optional
          texDesc.maxAnisotropy = 1;

          CUDA_CHECK(cuTexObjectCreate(&stream.cudaChromaTexObject, &chromaResourceDescriptor, &texDesc, /*resourceViewDescriptor=*/ nullptr));
        }
      } catch (const std::exception& ex) {
        printf("ArgusCameraMock: During CUDA resource setup for camera %zu: %s\n", cameraIdx, ex.what());
      }
    } // haveImageData

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

  m_previousFrameReadTime = now;
  return true;
}

CUtexObject ArgusCameraMock::cudaLumaTexObject(size_t sensorIdx) const {
  return m_streamData[sensorIdx].cudaLumaTexObject;
}

CUtexObject ArgusCameraMock::cudaChromaTexObject(size_t sensorIdx) const {
  return m_streamData[sensorIdx].cudaChromaTexObject;
}
