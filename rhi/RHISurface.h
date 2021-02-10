#pragma once
#include "rhi/RHIObject.h"
#include <glm/glm.hpp>

// Forward decl borrowed from <cuda.h>
typedef struct CUgraphicsResource_st *CUgraphicsResource; /**< CUDA graphics interop resource */

enum RHISamplerWrapMode : unsigned char {
  kWrapClamp,
  kWrapRepeat,
};

enum RHISamplerFilterMode : unsigned char {
  kFilterNearest,
  kFilterLinear,
  kFilterMipLinear,
};

struct RHISamplerDescriptor {
  RHISamplerDescriptor() : wrapModeU(kWrapClamp), wrapModeV(kWrapClamp), filter(kFilterNearest), maxAnisotropy(1) {}

  RHISamplerWrapMode wrapModeU, wrapModeV;
  RHISamplerFilterMode filter;
  uint8_t maxAnisotropy;
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
  kSurfaceFormat_R32f,
  kSurfaceFormat_RG32f,
  kSurfaceFormat_RGB16s,
  kSurfaceFormat_RGBA16s,

  kSurfaceFormat_Depth32f,
  kSurfaceFormat_Depth32f_Stencil8,
  kSurfaceFormat_Stencil8
};

size_t rhiSurfaceFormatSize(RHISurfaceFormat); // bytes per pixel
bool rhiSurfaceFormatHasDepth(RHISurfaceFormat);
bool rhiSurfaceFormatHasStencil(RHISurfaceFormat);

struct RHISurfaceDescriptor {
  RHISurfaceDescriptor(RHISurfaceFormat format_ = kSurfaceFormat_Invalid, uint8_t samples_ = 1) : format(format_), samples(samples_), layers(1), createArray(false), createMips(false) {}

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

  virtual CUgraphicsResource& cuGraphicsResource() const = 0;

  bool isMultisampled() const { return samples() > 1; }
  bool hasMipLevels() const { return mipLevels() > 1; }
  glm::vec2 dimensions() const { return glm::vec2(width(), height()); }
  glm::vec3 dimensions3() const { return glm::vec3(width(), height(), depth()); }
  float aspectRatio() const { return static_cast<float>(width()) / static_cast<float>(height()); }
};

class RHISampler : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHISampler> ptr;
  virtual ~RHISampler();

};

