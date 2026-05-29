#pragma once
#include "rhi/RHIObject.h"
#include <glm/glm.hpp>

// Forward decl borrowed from <cuda.h>
typedef unsigned long long cudaSurfaceObject_t;
typedef unsigned long long cudaTextureObject_t;
typedef struct cudaArray* cudaArray_t;

enum RHISamplerWrapMode : unsigned char {
  kWrapClamp,
  kWrapRepeat,
};

enum RHISamplerFilterMode : unsigned char {
  kFilterNearest,
  kFilterLinear,
  kFilterMipLinear,
};

enum RHIYcbcrModel : unsigned char {
  kYcbcrModelRGBIdentity,
  kYcbcrModelYcbcrIdentity,
  kYcbcrModelYcbcr709,
  kYcbcrModelYcbcr601,
  kYcbcrModelYcbcr2020,
};

enum RHIYcbcrRange : unsigned char {
  kYcbcrRangeFull,
  kYcbcrRangeNarrow,
};

enum RHIChromaLocation : unsigned char {
  kChromaLocationCositedEven,
  kChromaLocationMidpoint,
};

// Describes a YUV->RGB conversion to be applied at sample time. Attached to
// an RHISamplerDescriptor when the sampler will be used with a YUV-format
// surface (e.g. ArgusCamera output). The VK backend translates this into a
// VkSamplerYcbcrConversion; the GL backend ignores it (samplerExternalOES on
// GL handles equivalent conversion internally and was never user-configurable).
//
// sourceFormat names which YUV-plane layout the producer writes. Only NV12 is
// supported today; extend on demand.
struct RHIYcbcrConversionDescriptor {
  enum SourceFormat : unsigned char {
    kSourceFormatNone, // sampler is not ycbcr-aware (default)
    kSourceFormatNV12, // NV12 (Y + interleaved UV, 4:2:0). Maps to VK_FORMAT_G8_B8R8_2PLANE_420_UNORM.
  };

  RHIYcbcrConversionDescriptor() {}

  SourceFormat sourceFormat = kSourceFormatNone;
  RHIYcbcrModel model = kYcbcrModelYcbcr709;
  RHIYcbcrRange range = kYcbcrRangeNarrow;
  RHIChromaLocation xChromaOffset = kChromaLocationCositedEven;
  RHIChromaLocation yChromaOffset = kChromaLocationCositedEven;
  RHISamplerFilterMode chromaFilter = kFilterLinear;

  bool isEnabled() const { return sourceFormat != kSourceFormatNone; }
};

struct RHISamplerDescriptor {
  RHISamplerDescriptor() :
    wrapModeU(kWrapClamp),
    wrapModeV(kWrapClamp),
    filter(kFilterNearest),
    maxAnisotropy(1) {}

  RHISamplerWrapMode wrapModeU, wrapModeV;
  RHISamplerFilterMode filter;
  uint8_t maxAnisotropy;

  // When set, the sampler does YUV->RGB conversion at sample time. Samplers
  // configured this way carry implementation constraints in some backends
  // (e.g. VK requires them to be specified as immutable samplers in the
  // descriptor set layout — see RHIShaderDescriptor::setImmutableSampler).
  RHIYcbcrConversionDescriptor ycbcrConversion;
};

enum RHISurfaceFormat : unsigned char {
  kSurfaceFormat_Invalid,

  kSurfaceFormat_sRGB8_A8,
  kSurfaceFormat_RGBA8,
  kSurfaceFormat_RGB16f,
  kSurfaceFormat_RGBA16f,
  kSurfaceFormat_RGB10_A2,
  kSurfaceFormat_R8,
  kSurfaceFormat_R16,
  kSurfaceFormat_R16f,
  kSurfaceFormat_RG16,
  kSurfaceFormat_R32f,
  kSurfaceFormat_RG32f,
  kSurfaceFormat_RGB16s,
  kSurfaceFormat_RGBA16s,

  kSurfaceFormat_R8i,
  kSurfaceFormat_R8ui,
  kSurfaceFormat_R16i,
  kSurfaceFormat_R16ui,
  kSurfaceFormat_R32i,
  kSurfaceFormat_R32ui,

  kSurfaceFormat_RG8i,
  kSurfaceFormat_RG8ui,
  kSurfaceFormat_RG16i,
  kSurfaceFormat_RG16ui,
  kSurfaceFormat_RG32i,
  kSurfaceFormat_RG32ui,

  kSurfaceFormat_RGBA8i,
  kSurfaceFormat_RGBA8ui,
  kSurfaceFormat_RGBA16i,
  kSurfaceFormat_RGBA16ui,
  kSurfaceFormat_RGBA32i,
  kSurfaceFormat_RGBA32ui,

  kSurfaceFormat_Depth16,
  kSurfaceFormat_Depth32f,
  kSurfaceFormat_Depth32f_Stencil8,
  kSurfaceFormat_Stencil8
};

size_t rhiSurfaceFormatSize(RHISurfaceFormat); // bytes per pixel
bool rhiSurfaceFormatHasDepth(RHISurfaceFormat);
bool rhiSurfaceFormatHasStencil(RHISurfaceFormat);

struct RHISurfaceDescriptor {
  RHISurfaceDescriptor(RHISurfaceFormat format_ = kSurfaceFormat_Invalid, uint8_t samples_ = 1) :
    format(format_),
    samples(samples_),
    layers(1),
    createArray(false),
    createMips(false) {}

  static RHISurfaceDescriptor arrayDescriptor(RHISurfaceFormat format_, uint8_t layers_) {
    RHISurfaceDescriptor res(format_);
    res.createArray = true;
    res.layers = layers_;
    return res;
  }

  static RHISurfaceDescriptor mipDescriptor(RHISurfaceFormat format_) {
    RHISurfaceDescriptor res(format_);
    res.createMips = true;
    return res;
  }

  RHISurfaceFormat format;
  uint8_t samples, layers;
  bool createArray;
  bool createMips;
};

class RHISurface : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHISurface> ptr;
  virtual ~RHISurface();

  // use flags
  virtual bool isValidRenderTarget() const = 0;
  virtual bool isValidTextureSource() const = 0;

  virtual RHISurfaceFormat format() const = 0;

  virtual uint32_t width() const = 0;
  virtual uint32_t height() const = 0;
  virtual uint32_t depth() const = 0;
  virtual uint32_t layers() const = 0;
  virtual uint32_t samples() const = 0;
  virtual uint32_t mipLevels() const = 0;
  virtual bool isArray() const = 0;

  bool isMultisampled() const { return samples() > 1; }
  bool hasMipLevels() const { return mipLevels() > 1; }
  glm::vec2 dimensions() const { return glm::vec2(width(), height()); }
  glm::vec3 dimensions3() const { return glm::vec3(width(), height(), depth()); }
  float aspectRatio() const { return static_cast<float>(width()) / static_cast<float>(height()); }

  // CUDA Interop surface support
  virtual bool isInteropSurface() const;
  // cudaArray() is stable across the surface's lifetime.
  virtual cudaArray_t cudaArray() const;
};

class RHISampler : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHISampler> ptr;
  virtual ~RHISampler();
};
