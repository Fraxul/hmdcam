#pragma once
#include <stddef.h>
#include "rhi/RHISurface.h"
#include <cuda.h>
#include <opencv2/core/cuda.hpp>

class ICameraProvider {
public:
  virtual ~ICameraProvider() {}

  virtual size_t streamCount() const = 0;
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const = 0;
  // Luma and Chroma CUtexObjects use linear filtering, clamp-to-edge, and normalized (0-1) coordinates.
  virtual CUtexObject cudaLumaTexObject(size_t sensorIndex) const = 0;
  virtual CUtexObject cudaChromaTexObject(size_t sensorIndex) const = 0;
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t sensorIndex) = 0;
  virtual bool fillCudaMemcpy2DForStreamSource(CUDA_MEMCPY2D& outCopyDescriptor, size_t sensorIndex, bool fromChromaPlane) const = 0;
  virtual unsigned int streamWidth() const = 0;
  virtual unsigned int streamHeight() const = 0;
  virtual bool isStreamFailed(size_t sensorIndex) const = 0;

  // Sampler that should be bound when sampling rgbTexture() in a render
  // pipeline. Non-null providers (e.g. Tegra ArgusCamera on the VK backend)
  // return a ycbcr-aware sampler whose VkSamplerYcbcrConversion must be
  // baked into the consuming pipeline as an immutable-sampler binding
  // (see RHIShaderDescriptor::setImmutableSamplerBinding). Mock / GL-only
  // providers return null, and the consuming pipeline uses a regular
  // pushable sampler.
  virtual RHISampler::ptr cameraSampler() const { return RHISampler::ptr(); }

  // Scale factor that consumers must multiply their normalized X-tex-coord
  // by when sampling rgbTexture(). Needed when the underlying VkImage is
  // wider than the valid Argus output region (NvBuf pitch-pad workaround on
  // Tegra). Default 1.0 means no crop. The same factor applies to every
  // camera stream in the provider (they all share resolution).
  virtual float cameraTexCoordCropX() const { return 1.0f; }

  // Nanosecond timestamp for the current frame (relative to currentTimeNs())
  virtual uint64_t frameTimestamp() const = 0;

  struct FrameMetadata {
    uint64_t sensorTimestamp = 0;
    uint64_t frameDurationNs = 0;
    uint64_t sensorExposureTimeNs = 0;
    uint32_t sensorSensitivityISO = 0;
    float ispDigitalGain = 0.0f;
    float sensorAnalogGain = 0.0f;
    float sceneLux = 0.0f;
    uint32_t awbCct = 0;
  };

  // Metadata accessors for the current frame
  // Start timestamp for the sensor capture, in nanoseconds. TSC timestamp, referenced to currentTimeNs()
  uint64_t frameSensorTimestamp(size_t sensorIndex) const { return m_frameMetadata[sensorIndex].sensorTimestamp; }
  const FrameMetadata& frameMetadata(size_t sensorIndex) const { return m_frameMetadata[sensorIndex]; }

protected:
  // Per-stream per-frame metadata, populated for each frame in readFrame()
  std::vector<FrameMetadata> m_frameMetadata;
};

class NullCameraProvider : public ICameraProvider {
public:
  NullCameraProvider(size_t streamCount_, unsigned int w_ = 1920, unsigned int h_ = 1080) :
    m_streamCount(streamCount_),
    m_streamWidth(w_),
    m_streamHeight(h_) {

    m_frameMetadata.resize(m_streamCount);
  }
  virtual ~NullCameraProvider() {}

  virtual size_t streamCount() const { return m_streamCount; }
  virtual RHISurface::ptr rgbTexture(size_t sensorIndex) const { return RHISurface::ptr(); }
  virtual CUtexObject cudaLumaTexObject(size_t sensorIndex) const { return 0; }
  virtual CUtexObject cudaChromaTexObject(size_t sensorIndex) const { return 0; }
  virtual cv::cuda::GpuMat gpuMatGreyscale(size_t sensorIndex) { return cv::cuda::GpuMat(); }
  virtual bool fillCudaMemcpy2DForStreamSource(CUDA_MEMCPY2D& outCopyDescriptor, size_t sensorIndex, bool fromChromaPlane) const { return false; }
  virtual unsigned int streamWidth() const { return m_streamWidth; }
  virtual unsigned int streamHeight() const { return m_streamHeight; }
  virtual bool isStreamFailed(size_t sensorIndex) const { return false; }
  virtual uint64_t frameTimestamp() const { return 0; }

protected:
  size_t m_streamCount;
  unsigned int m_streamWidth, m_streamHeight;
};
