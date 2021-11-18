#include "rhi/gl/RHISurfaceGL.h"
#include "rhi/FxMath.h"
#include <algorithm>
#include <cuda.h>
#include <cudaGL.h>
#include "rhi/cuda/CudaUtil.h"

GLenum RHISurfaceFormatToGL(RHISurfaceFormat format) {
  switch (format) {
    case kSurfaceFormat_Invalid:
    default:
      assert(false && "RHISurfaceFormatToGL: kSurfaceFormat_Invalid");
      return GL_NONE;

    case kSurfaceFormat_sRGB8_A8: return GL_SRGB8_ALPHA8;
    case kSurfaceFormat_RGBA8: return GL_RGBA8;
    case kSurfaceFormat_RGB16f: return GL_RGB16F;
    case kSurfaceFormat_RGBA16f: return GL_RGBA16F;

    case kSurfaceFormat_RGB10_A2: return GL_RGB10_A2;
    case kSurfaceFormat_R8: return GL_R8;
#ifndef GL_R16
#define GL_R16 GL_R16_EXT
#endif
    case kSurfaceFormat_R16: return GL_R16;
    case kSurfaceFormat_R16f: return GL_R16F;
    case kSurfaceFormat_R32f: return GL_R32F;
    case kSurfaceFormat_RG32f: return GL_RG32F;
#ifndef GL_RGB16_SNORM_EXT
#define GL_RGB16_SNORM_EXT GL_RGB16_SNORM
#endif
    case kSurfaceFormat_RGB16s: return GL_RGB16_SNORM_EXT;

#ifndef GL_RGBA16_SNORM_EXT
#define GL_RGBA16_SNORM_EXT GL_RGBA16_SNORM
#endif
    case kSurfaceFormat_RGBA16s: return GL_RGBA16_SNORM_EXT;

    case kSurfaceFormat_R8i: return GL_R8I;
    case kSurfaceFormat_R8ui: return GL_R8UI;
    case kSurfaceFormat_R16i: return GL_R16I;
    case kSurfaceFormat_R16ui: return GL_R16UI;
    case kSurfaceFormat_R32i: return GL_R32I;
    case kSurfaceFormat_R32ui: return GL_R32UI;

    case kSurfaceFormat_RG8i: return GL_RG8I;
    case kSurfaceFormat_RG8ui: return GL_RG8UI;
    case kSurfaceFormat_RG16i: return GL_RG16I;
    case kSurfaceFormat_RG16ui: return GL_RG16UI;
    case kSurfaceFormat_RG32i: return GL_RG32I;
    case kSurfaceFormat_RG32ui: return GL_RG32UI;

    case kSurfaceFormat_RGBA8i: return GL_RGBA8I;
    case kSurfaceFormat_RGBA8ui: return GL_RGBA8UI;
    case kSurfaceFormat_RGBA16i: return GL_RGBA16I;
    case kSurfaceFormat_RGBA16ui: return GL_RGBA16UI;
    case kSurfaceFormat_RGBA32i: return GL_RGBA32I;
    case kSurfaceFormat_RGBA32ui: return GL_RGBA32UI;

    case kSurfaceFormat_Depth16: return GL_DEPTH_COMPONENT16;
    case kSurfaceFormat_Depth32f: return GL_DEPTH_COMPONENT32F;
    case kSurfaceFormat_Depth32f_Stencil8: return GL_DEPTH32F_STENCIL8;
    case kSurfaceFormat_Stencil8: return GL_STENCIL_INDEX8;
  };
}

static GLint rhiSamplerWrapModeToGL(RHISamplerWrapMode mode) {
  switch (mode) {
    case kWrapClamp: return GL_CLAMP_TO_BORDER;
    case kWrapRepeat: return GL_REPEAT;
    default:
      assert(false && "rhiSamplerWrapModeToGL: unknown enum");
      return GL_NONE;
  };
}

static GLint rhiSamplerFilterModeToGL(RHISamplerFilterMode mode) {
  switch (mode) {
    case kFilterNearest: return GL_NEAREST;
    case kFilterLinear: return GL_LINEAR;
    case kFilterMipLinear: return GL_LINEAR_MIPMAP_LINEAR;
    default:
      assert(false && "rhiSamplerFilterModeToGL: unknown enum");
      return GL_NONE;
  };
}

RHISamplerGL::RHISamplerGL(const RHISamplerDescriptor& descriptor) {
  glGenSamplers(1, &m_glId);

  GLint wrapS = rhiSamplerWrapModeToGL(descriptor.wrapModeU);
  GLint wrapT = rhiSamplerWrapModeToGL(descriptor.wrapModeV);
  GL(glSamplerParameteriv(m_glId, GL_TEXTURE_WRAP_S, &wrapS));
  GL(glSamplerParameteriv(m_glId, GL_TEXTURE_WRAP_T, &wrapT));

  GLint filter = rhiSamplerFilterModeToGL(descriptor.filter);
  GL(glSamplerParameteriv(m_glId, GL_TEXTURE_MIN_FILTER, &filter));
  GL(glSamplerParameteriv(m_glId, GL_TEXTURE_MAG_FILTER, &filter));

  GLint aniso = descriptor.maxAnisotropy;
  GL(glSamplerParameteriv(m_glId, GL_TEXTURE_MAX_ANISOTROPY_EXT, &aniso));
}

RHISamplerGL::~RHISamplerGL() {
  glDeleteSamplers(1, &m_glId);
}

RHISurfaceGL::RHISurfaceGL() : m_glId(0), m_glTarget(0), m_glInternalFormat(0), m_width(0), m_height(0), m_depth(0), m_layers(1), m_samples(1), m_levels(1), m_rhiFormat(kSurfaceFormat_Invalid), m_isArrayTexture(false), m_cuGraphicsResource(NULL) {

}

RHISurfaceGL::~RHISurfaceGL() {
  if (m_glId) {
    if (m_cuGraphicsResource)
      cuGraphicsUnregisterResource(m_cuGraphicsResource);

    if (m_glTarget == GL_RENDERBUFFER) {
      glDeleteRenderbuffers(1, &m_glId);
    } else {
      glDeleteTextures(1, &m_glId);
    }
  }
}

bool RHISurfaceGL::isValidRenderTarget() const {
  return true;
}

bool RHISurfaceGL::isValidTextureSource() const {
  return isGLTexture();
}

/*static*/ RHISurfaceGL* RHISurfaceGL::newRenderbuffer2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor& descriptor) {
  RHISurfaceGL* srf = new RHISurfaceGL();

  glGenRenderbuffers(1, &srf->m_glId);
  srf->m_glTarget = GL_RENDERBUFFER;
  srf->m_glInternalFormat = RHISurfaceFormatToGL(descriptor.format);
  srf->m_width = width;
  srf->m_height = height;
  srf->m_depth = 1;
  srf->m_layers = 1;
  srf->m_samples = descriptor.samples;
  srf->m_levels = 1;
  srf->m_rhiFormat = descriptor.format;
  srf->m_isArrayTexture = false;

  glBindRenderbuffer(GL_RENDERBUFFER, srf->m_glId);
  if (srf->m_samples > 1) {
    GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, srf->m_samples, srf->m_glInternalFormat, srf->m_width, srf->m_height));
  } else {
    GL(glRenderbufferStorage(GL_RENDERBUFFER, srf->m_glInternalFormat, srf->m_width, srf->m_height));
  }

  return srf;
}

/*static*/ RHISurfaceGL* RHISurfaceGL::newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor& descriptor) {
  RHISurfaceGL* tex = new RHISurfaceGL();

  GL(glGenTextures(1, &tex->m_glId));
  // m_glTarget set later
  tex->m_rhiFormat = descriptor.format;
  tex->m_glInternalFormat = RHISurfaceFormatToGL(descriptor.format);
  tex->m_width = width;
  tex->m_height = height;
  tex->m_depth = 1;
  tex->m_layers = descriptor.createArray ? descriptor.layers : 1;
  tex->m_samples = descriptor.samples;
  if (descriptor.createMips) {
    uint32_t maxDim = std::max<uint32_t>(width, height);
    tex->m_levels = floorLogTwo(maxDim) + 1;
  } else {
    tex->m_levels = 1;
  }
  tex->m_isArrayTexture = descriptor.createArray;

  if (tex->m_samples > 1) {
    // multi-sampled
    assert(tex->m_levels == 1 && "Incompatible texture format options: mipmaps not allowed on multisampled textures");
    if (tex->m_isArrayTexture) {
      tex->m_glTarget = GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
      GL(glBindTexture(tex->m_glTarget, tex->m_glId));
      GL(glTexStorage3DMultisample(tex->m_glTarget, tex->m_samples, tex->m_glInternalFormat, tex->m_width, tex->m_height, tex->m_layers, GL_TRUE));
    } else {
      tex->m_glTarget = GL_TEXTURE_2D_MULTISAMPLE;
      GL(glBindTexture(tex->m_glTarget, tex->m_glId));
      GL(glTexStorage2DMultisample(tex->m_glTarget, tex->m_samples, tex->m_glInternalFormat, tex->m_width, tex->m_height, GL_TRUE));
    }
  } else {
    // single-sampled
    if (tex->m_isArrayTexture) {
      tex->m_glTarget = GL_TEXTURE_2D_ARRAY;
      GL(glBindTexture(tex->m_glTarget, tex->m_glId));
      GL(glTexStorage3D(tex->m_glTarget, tex->m_levels, tex->m_glInternalFormat, tex->m_width, tex->m_height, tex->m_layers));
    } else {
      tex->m_glTarget = GL_TEXTURE_2D;
      GL(glBindTexture(tex->m_glTarget, tex->m_glId));
      GL(glTexStorage2D(tex->m_glTarget, tex->m_levels, tex->m_glInternalFormat, tex->m_width, tex->m_height));
    }

    GL(glTexParameteri(tex->m_glTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(tex->m_glTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(tex->m_glTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL(glTexParameteri(tex->m_glTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
  }

  return tex;
}

/*static*/ RHISurfaceGL* RHISurfaceGL::newTexture3D(uint32_t width, uint32_t height, uint32_t depth, const RHISurfaceDescriptor& descriptor) {
  assert(descriptor.samples == 1 && "Incompatible texture format options: 3d multisample textures are not supported.");
  assert(descriptor.layers == 1 && descriptor.createArray == false && "Incompatible texture format options: 3d array textures are not supported.");
  RHISurfaceGL* tex = new RHISurfaceGL();

  GL(glGenTextures(1, &tex->m_glId));
  tex->m_glTarget = GL_TEXTURE_3D;
  tex->m_glInternalFormat = RHISurfaceFormatToGL(descriptor.format);
  tex->m_rhiFormat = descriptor.format;
  tex->m_width = width;
  tex->m_height = height;
  tex->m_depth = depth;
  tex->m_layers = 1;
  tex->m_samples = 1;
  if (descriptor.createMips) {
    uint32_t maxDim = std::max<uint32_t>(std::max<uint32_t>(width, height), depth);
    tex->m_levels = floorLogTwo(maxDim) + 1;
  } else {
    tex->m_levels = 1;
  }
  GL(glBindTexture(tex->m_glTarget, tex->m_glId));
  GL(glTexStorage3D(tex->m_glTarget, tex->m_levels, tex->m_glInternalFormat, tex->m_width, tex->m_height, tex->m_depth));

  GL(glTexParameteri(tex->m_glTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
  GL(glTexParameteri(tex->m_glTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
  GL(glTexParameteri(tex->m_glTarget, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
  GL(glTexParameteri(tex->m_glTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
  GL(glTexParameteri(tex->m_glTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST));

  return tex;
}

RHISurfaceFormat RHISurfaceGL::format() const {
  return m_rhiFormat;
}

uint32_t RHISurfaceGL::width() const {
  return m_width;
}

uint32_t RHISurfaceGL::height() const {
  return m_height;
}

uint32_t RHISurfaceGL::depth() const {
  return m_depth;
}

uint32_t RHISurfaceGL::layers() const {
  return m_layers;
}

uint32_t RHISurfaceGL::samples() const {
  return m_samples;
}

uint32_t RHISurfaceGL::mipLevels() const {
  return m_levels;
}

bool RHISurfaceGL::isArray() const {
  return m_isArrayTexture;
}

CUgraphicsResource& RHISurfaceGL::cuGraphicsResource() const {
  if (!m_cuGraphicsResource) {
    CUDA_CHECK(cuGraphicsGLRegisterImage(&m_cuGraphicsResource, glId(), glTarget(), CU_GRAPHICS_REGISTER_FLAGS_NONE));
  }

  return m_cuGraphicsResource;
}

