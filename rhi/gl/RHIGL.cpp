#include "rhi/RHIResources.h"
#include "rhi/gl/RHIGL.h"
#include "rhi/gl/RHIBlendStateGL.h"
#include "rhi/gl/RHIBufferGL.h"
#include "rhi/gl/RHIComputePipelineGL.h"
#include "rhi/gl/RHIQueryGL.h"
#include "rhi/gl/RHIRenderPipelineGL.h"
#include "rhi/gl/RHIRenderTargetGL.h"
#include "rhi/gl/RHIShaderGL.h"
#include "rhi/gl/RHISurfaceGL.h"
#include "rhi/gl/RHIWindowRenderTargetGL.h"
#include <glm/gtc/type_ptr.hpp>

static /*CVar*/ bool rhi_gl_forceBlitEmulation = false;

static bool s_isFirstRHIGLInit = true;

void initRHIGL() {
  initRHI(new RHIGL());
}

RHIGL::RHIGL() : m_clearColor(glm::vec4(0.0f)), m_clearDepth(1.0f), m_clearStencil(0), m_uniformBufferOffsetAlignment(256), m_maxMultisampleSamples(1), m_inComputePass(false),
  m_currentCullState(kCullDisabled),
  m_currentDepthBiasSlopeScale(0.0f),
  m_currentDepthBiasConstant(0.0f) {

  // set global options
  if (s_isFirstRHIGLInit) {
    s_ndcZNearIsNegativeOne = true;
    s_allowsLayerSelectionFromVertexShader = false; // (GLEW_AMD_vertex_shader_layer);
    // printf("RHIGL: allowsLayerSelectionFromVertexShader = %d\n", s_allowsLayerSelectionFromVertexShader);

  }

  GLint t;

  {
    glGetIntegerv(GL_MAX_COLOR_TEXTURE_SAMPLES, &t);
    m_maxMultisampleSamples = t;
    // These should be the same, but it doesn't hurt to check.
    glGetIntegerv(GL_MAX_DEPTH_TEXTURE_SAMPLES, &t);
    m_maxMultisampleSamples = std::min<uint32_t>(t, m_maxMultisampleSamples);
    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &t);
    m_uniformBufferOffsetAlignment = t;

    if (s_isFirstRHIGLInit) {
      printf("RHIGL: maxMultisampleSamples = %d\n", m_maxMultisampleSamples);
      printf("RHIGL: uniformBufferOffsetAlignment = %d\n", m_uniformBufferOffsetAlignment);
    }
  }

  s_isFirstRHIGLInit = false;

  m_immediateScratchBufferSize = 16 * 1024 * 1024;
  m_immediateScratchBufferWriteOffset = 0;
  glGenBuffers(1, &m_immediateScratchBufferId);
  glBindBuffer(GL_UNIFORM_BUFFER, m_immediateScratchBufferId);
#ifndef GL_MAP_PERSISTENT_BIT_EXT
#define GL_MAP_PERSISTENT_BIT_EXT GL_MAP_PERSISTENT_BIT
#endif
#ifndef GL_MAP_COHERENT_BIT_EXT
#define GL_MAP_COHERENT_BIT_EXT GL_MAP_COHERENT_BIT
#endif
  GLenum mapFlags = GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_COHERENT_BIT_EXT;

#ifndef glBufferStorageEXT
#define glBufferStorageEXT glBufferStorage
#endif
  GL(glBufferStorageEXT(GL_UNIFORM_BUFFER, m_immediateScratchBufferSize, NULL, mapFlags));
  GL(m_immediateScratchBufferData = static_cast<char*>(glMapBufferRange(GL_UNIFORM_BUFFER, 0, m_immediateScratchBufferSize, mapFlags | GL_MAP_INVALIDATE_BUFFER_BIT)));
  if (!m_immediateScratchBufferData) {
    assert(false && "RHIGL: Unable to create and map scratch buffer");
  }
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
}

RHIGL::~RHIGL() {
  glBindBuffer(GL_UNIFORM_BUFFER, m_immediateScratchBufferId);
  glUnmapBuffer(GL_UNIFORM_BUFFER);
  glBindBuffer(GL_UNIFORM_BUFFER, 0);
  glDeleteBuffers(1, &m_immediateScratchBufferId);
}

RHIDepthStencilState::ptr RHIGL::compileDepthStencilState(const RHIDepthStencilStateDescriptor& descriptor) {
  return RHIDepthStencilState::ptr(new RHIDepthStencilStateGL(descriptor));
}

RHIRenderTarget::ptr RHIGL::compileRenderTarget(const RHIRenderTargetDescriptor& descriptor) {
  return RHIRenderTarget::ptr(RHIRenderTargetGL::newRenderTarget(descriptor));
}

RHISampler::ptr RHIGL::compileSampler(const RHISamplerDescriptor& descriptor) {
  return RHISampler::ptr(new RHISamplerGL(descriptor));
}

RHIBlendState::ptr RHIGL::compileBlendState(const RHIBlendStateDescriptor& descriptor) {
  return RHIBlendState::ptr(new RHIBlendStateGL(descriptor));
}

RHIShader::ptr RHIGL::internalCompileShader(const RHIShaderDescriptor& descriptor) {
  return RHIShader::ptr(new RHIShaderGL(descriptor));
}

RHIRenderPipeline::ptr RHIGL::internalCompileRenderPipeline(RHIShader::ptr shader, const RHIRenderPipelineDescriptor& descriptor) {
  return RHIRenderPipeline::ptr(new RHIRenderPipelineGL(static_cast<RHIShaderGL*>(shader.get()), descriptor));
}

RHIComputePipeline::ptr RHIGL::internalCompileComputePipeline(RHIShader::ptr shader) {
  return RHIComputePipeline::ptr(new RHIComputePipelineGL(static_cast<RHIShaderGL*>(shader.get())));
}

static GLenum RHIBufferUsageModeToGL(RHIBufferUsageMode mode) {
  switch (mode) {
    case kBufferUsageCPUWriteOnly:
      return GL_STATIC_DRAW;
    case kBufferUsageCPUReadback:
      return GL_DYNAMIC_READ;
    case kBufferUsageGPUPrivate:
      return GL_DYNAMIC_COPY;

    default:
      assert(false && "RHIBufferUsageModeToGL: unhandled enum");
      return GL_STATIC_DRAW;
  };
}

void RHIGL::loadBufferData(RHIBuffer::ptr buf, const void* data, size_t offset, size_t length) {
  RHIBufferGL* bufferGL = static_cast<RHIBufferGL*>(buf.get());
  assert(offset < bufferGL->size());
  if (!length) {
    length = bufferGL->size() - offset;
  }
  if (offset == 0 && length == bufferGL->size()) {
    bufferGL->bufferData(data, bufferGL->size());
  } else {
    bufferGL->bufferSubData(data, length, offset);
  }
}

RHIBuffer::ptr RHIGL::newBufferWithContents(const void* data, size_t size, RHIBufferUsageMode usageMode) {
  GLuint buffer;
  GL(glGenBuffers(1, &buffer));
  GL(glBindBuffer(GL_ARRAY_BUFFER, buffer));
  GL(glBufferData(GL_ARRAY_BUFFER, size, data, RHIBufferUsageModeToGL(usageMode)));
  return RHIBuffer::ptr(new RHIBufferGL(buffer, size, usageMode));
}

RHIBuffer::ptr RHIGL::newEmptyBuffer(size_t size, RHIBufferUsageMode usageMode) {
  return newBufferWithContents(NULL, size, usageMode);
}

RHIBuffer::ptr RHIGL::newUniformBufferWithContents(const void* data, size_t size) {
  GLuint buffer;
  GL(glGenBuffers(1, &buffer));
  GL(glBindBuffer(GL_ARRAY_BUFFER, buffer));
  GL(glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW));
  return RHIBuffer::ptr(new RHIBufferGL(buffer, size, kBufferUsageCPUWriteOnly));
}

void RHIGL::clearBuffer(RHIBuffer::ptr buffer) {
  RHIBufferGL* glBuf = static_cast<RHIBufferGL*>(buffer.get());
  GL(glBindBuffer(GL_ARRAY_BUFFER, glBuf->glId()));
  //GL(glClearBufferData(GL_ARRAY_BUFFER, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, NULL));
  assert(false && "RHIGL::clearBuffer: Not implemented on ES3");
}

RHISurface::ptr RHIGL::newTexture2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor& descriptor) {
  return RHISurface::ptr(RHISurfaceGL::newTexture2D(width, height, descriptor));
}

RHISurface::ptr RHIGL::newTexture3D(uint32_t width, uint32_t height, uint32_t depth, const RHISurfaceDescriptor& descriptor) {
  return RHISurface::ptr(RHISurfaceGL::newTexture3D(width, height, depth, descriptor));
}

RHISurface::ptr RHIGL::newRenderbuffer2D(uint32_t width, uint32_t height, const RHISurfaceDescriptor& descriptor) {
  return RHISurface::ptr(RHISurfaceGL::newRenderbuffer2D(width, height, descriptor));
}

RHISurface::ptr RHIGL::newHMDSwapTexture(uint32_t width, uint32_t height, const RHISurfaceDescriptor&) {
  assert(false && "RHIGL::newHMDSwapTexture: not implemented");
}

static void rhiVertexElementTypeToGLPackFormat(RHIVertexElementType rhiFormat, GLenum& unpackFormat, GLenum& unpackType) {
  switch (rhiFormat) {
    case kVertexElementTypeFloat1:
      unpackFormat = GL_RED; unpackType = GL_FLOAT; return;
    case kVertexElementTypeFloat2:
      unpackFormat = GL_RG; unpackType = GL_FLOAT; return;
    case kVertexElementTypeFloat3:
      unpackFormat = GL_RGB; unpackType = GL_FLOAT; return;
    case kVertexElementTypeFloat4:
      unpackFormat = GL_RGBA; unpackType = GL_FLOAT; return;
    case kVertexElementTypeHalf1:
      unpackFormat = GL_RED; unpackType = GL_HALF_FLOAT; return;
    case kVertexElementTypeHalf2:
      unpackFormat = GL_RG; unpackType = GL_HALF_FLOAT; return;
    case kVertexElementTypeHalf4:
      unpackFormat = GL_RGBA; unpackType = GL_HALF_FLOAT; return;
    case kVertexElementTypeUShort1:
    case kVertexElementTypeUShort1N:
      unpackFormat = GL_RED; unpackType = GL_UNSIGNED_SHORT; return;
    case kVertexElementTypeUShort2:
    case kVertexElementTypeUShort2N:
      unpackFormat = GL_RG; unpackType = GL_UNSIGNED_SHORT; return;
    case kVertexElementTypeUShort4:
    case kVertexElementTypeUShort4N:
      unpackFormat = GL_RGBA; unpackType = GL_UNSIGNED_SHORT; return;
    case kVertexElementTypeShort1:
    case kVertexElementTypeShort1N:
      unpackFormat = GL_RED; unpackType = GL_SHORT; return;
    case kVertexElementTypeShort2:
    case kVertexElementTypeShort2N:
      unpackFormat = GL_RG; unpackType = GL_SHORT; return;
    case kVertexElementTypeShort4:
    case kVertexElementTypeShort4N:
      unpackFormat = GL_RGBA; unpackType = GL_SHORT; return;
    case kVertexElementTypeUByte1:
    case kVertexElementTypeUByte1N:
      unpackFormat = GL_RED; unpackType = GL_UNSIGNED_BYTE; return;
    case kVertexElementTypeUByte2:
    case kVertexElementTypeUByte2N:
      unpackFormat = GL_RG; unpackType = GL_UNSIGNED_BYTE; return;
    case kVertexElementTypeUByte4:
    case kVertexElementTypeUByte4N:
      unpackFormat = GL_RGBA; unpackType = GL_UNSIGNED_BYTE; return;
    case kVertexElementTypeByte1:
    case kVertexElementTypeByte1N:
      unpackFormat = GL_RED; unpackType = GL_BYTE; return;
    case kVertexElementTypeByte2:
    case kVertexElementTypeByte2N:
      unpackFormat = GL_RG; unpackType = GL_BYTE; return;
    case kVertexElementTypeByte4:
    case kVertexElementTypeByte4N:
      unpackFormat = GL_RGBA; unpackType = GL_BYTE; return;
    case kVertexElementTypeUInt1:
      unpackFormat = GL_RED; unpackType = GL_UNSIGNED_INT; return;
    case kVertexElementTypeUInt2:
      unpackFormat = GL_RG; unpackType = GL_UNSIGNED_INT; return;
    case kVertexElementTypeUInt4:
      unpackFormat = GL_RGBA; unpackType = GL_UNSIGNED_INT; return;
    case kVertexElementTypeInt1:
      unpackFormat = GL_RED; unpackType = GL_INT; return;
    case kVertexElementTypeInt2:
      unpackFormat = GL_RG; unpackType = GL_INT; return;
    case kVertexElementTypeInt4:
      unpackFormat = GL_RGBA; unpackType = GL_INT; return;
    default:
      assert(false && "RHIGL rhiVertexElementTypeToGLPackFormat: unimplemented sourceDataFormat");
  }
}

bool isIntegerInternalFormat(GLenum internalFormat) {
  switch (internalFormat) {
    case GL_R8I:
    case GL_R8UI:
    case GL_R16I:
    case GL_R16UI:
    case GL_R32I:
    case GL_R32UI:

    case GL_RG8I:
    case GL_RG8UI:
    case GL_RG16I:
    case GL_RG16UI:
    case GL_RG32I:
    case GL_RG32UI:

    case GL_RGBA8I:
    case GL_RGBA8UI:
    case GL_RGBA16I:
    case GL_RGBA16UI:
    case GL_RGBA32I:
    case GL_RGBA32UI:
      return true;

    default:
      return false;
  }
}

GLenum toIntegerUnpackFormat(GLenum unpackFormat) {
  switch (unpackFormat) {
    case GL_RED: return GL_RED_INTEGER;
    case GL_RG: return GL_RG_INTEGER;
    case GL_RGB: return GL_RGB_INTEGER;
    case GL_RGBA: return GL_RGBA_INTEGER;
    default:
      fprintf(stderr, "toIntegerUnpackFormat(): unhandled case 0x%x", unpackFormat);
      abort();
      return 0;
  }
}


void RHIGL::loadTextureData(RHISurface::ptr texture, RHIVertexElementType sourceDataFormat, const void* sourceData) {
  RHISurfaceGL* tex = static_cast<RHISurfaceGL*>(texture.get());

  GLenum unpackFormat, unpackType;
  rhiVertexElementTypeToGLPackFormat(sourceDataFormat, unpackFormat, unpackType);

  if (isIntegerInternalFormat(tex->glInternalFormat())) {
    unpackFormat = toIntegerUnpackFormat(unpackFormat);
  }

  assert(tex->glTarget() == GL_TEXTURE_2D); // only handled case for now

  GL(glBindTexture(tex->glTarget(), tex->glId()));
  GL(glTexSubImage2D(tex->glTarget(), 0, 0, 0, tex->width(), tex->height(), unpackFormat, unpackType, sourceData));

  if (tex->mipLevels() > 1) {
    glGenerateMipmap(tex->glTarget());
    glTexParameteri(tex->glTarget(), GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
  }

}

void RHIGL::generateTextureMips(RHISurface::ptr texture) {
  RHISurfaceGL* glTex = static_cast<RHISurfaceGL*>(texture.get());

  assert(glTex->glTarget() == GL_TEXTURE_2D ||
         glTex->glTarget() == GL_TEXTURE_3D ||
         glTex->glTarget() == GL_TEXTURE_2D_ARRAY ||
         glTex->glTarget() == GL_TEXTURE_CUBE_MAP ||
         glTex->glTarget() == GL_TEXTURE_CUBE_MAP_ARRAY);

  GL(glBindTexture(glTex->glTarget(), glTex->glId()));
  GL(glGenerateMipmap(glTex->glTarget()));
}

static GLuint generateSingleLayerFramebufferForTexture(RHISurfaceGL* glTex, uint8_t layer) {
  GLuint fbo = 0;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  if (glTex->isArray()) {
    glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, glTex->glId(), 0, layer);
  } else {
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, glTex->glId(), 0);
  }
  assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
  return fbo;
}

void RHIGL::readbackTexture(RHISurface::ptr texture, uint8_t layer, RHIVertexElementType dataFormat, void* outData) {
  assert(!m_activeRenderTarget && "RHIGL::readbackTexture: can't do this during a render pass");

  RHISurfaceGL* glTex = static_cast<RHISurfaceGL*>(texture.get());

  GLenum unpackFormat, unpackType;
  rhiVertexElementTypeToGLPackFormat(dataFormat, unpackFormat, unpackType);

  GLuint fbo = generateSingleLayerFramebufferForTexture(glTex, layer);
  GL(glReadPixels(0, 0, texture->width(), texture->height(), unpackFormat, unpackType, outData));
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
}

void RHIGL::fillOpenVRTextureStruct(RHISurface::ptr, vr::Texture_t*) {
  assert(false && "RHIGL::fillOpenVRTextureStruct(): not implemented");
}

void RHIGL::setClearColor(const glm::vec4 color) {
  m_clearColor = color;
}

void RHIGL::setClearDepth(float depth) {
  m_clearDepth = depth;
}

void RHIGL::setClearStencil(uint8_t stencil) {
  m_clearStencil = stencil;
}

void RHIGL::beginRenderPass(RHIRenderTarget::ptr renderTarget, RHIRenderTargetLoadAction colorLoadAction, RHIRenderTargetLoadAction depthLoadAction, RHIRenderTargetLoadAction stencilLoadAction) {
  assert(!m_inComputePass && "beginRenderPass: currently encoding a compute pass");
  assert(m_activeRenderTarget.get() == NULL && "beginRenderPass: endRenderPass was not called for the previous pass");
  assert(colorLoadAction != kLoadUnspecified && "beginRenderPass: kLoadUnspecified is not a valid load action type");
  if (depthLoadAction == kLoadUnspecified)
    depthLoadAction = colorLoadAction;
  if (stencilLoadAction == kLoadUnspecified)
    stencilLoadAction = depthLoadAction;

  m_activeRenderTarget = static_cast<RHIRenderTargetGL*>(renderTarget.get());

  GL(glBindFramebuffer(GL_FRAMEBUFFER, m_activeRenderTarget->glFramebufferId()));
  GL(glViewport(0, 0, m_activeRenderTarget->width(), m_activeRenderTarget->height()));

  static GLenum buffers[] = {
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_ATTACHMENT1,
    GL_COLOR_ATTACHMENT2,
    GL_COLOR_ATTACHMENT3,
    GL_COLOR_ATTACHMENT4,
    GL_COLOR_ATTACHMENT5,
    GL_COLOR_ATTACHMENT6,
    GL_COLOR_ATTACHMENT7,
    GL_COLOR_ATTACHMENT8,
    GL_COLOR_ATTACHMENT9,
    GL_COLOR_ATTACHMENT10,
    GL_COLOR_ATTACHMENT11,
    GL_COLOR_ATTACHMENT12,
    GL_COLOR_ATTACHMENT13,
    GL_COLOR_ATTACHMENT14,
    GL_COLOR_ATTACHMENT15 };

  if (m_activeRenderTarget->glFramebufferId() == 0) {
    // special case for window RT
    GLenum backBuffer = GL_BACK;
    GL(glDrawBuffers(1, &backBuffer));
    GL(glReadBuffer(GL_BACK));
  } else if (m_activeRenderTarget->colorTargetCount()) {
    // this just picks the first (m_colorBuffersCount) buffers from the array which contains all of the COLOR_ATTACHMENT enums in order
    GL(glDrawBuffers(m_activeRenderTarget->colorTargetCount(), buffers));
  }

  // disable all color/depth/stencil masking before clears.
  // masking is part of the render pipeline state so will be reset when that is next bound
  GL(glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE));
  GL(glDepthMask(GL_TRUE));
  GL(glStencilMask(0xffffffff));

  // execute clears
  uint32_t clearBits = 0;
  if (m_activeRenderTarget->hasColorTarget() && (colorLoadAction == kLoadClear)) {
    glClearColor(m_clearColor[0], m_clearColor[1], m_clearColor[2], m_clearColor[3]);
    clearBits |= GL_COLOR_BUFFER_BIT;
  }

  if (m_activeRenderTarget->hasDepthStencilTarget()) {
    if (depthLoadAction == kLoadClear) {
      glClearDepthf(m_clearDepth);
      clearBits |= GL_DEPTH_BUFFER_BIT;
    }
    if (stencilLoadAction == kLoadClear) {
      glClearStencil(m_clearStencil);
      clearBits |= GL_STENCIL_BUFFER_BIT;
    }
  }

  if (clearBits) {
    GL(glClear(clearBits));
  }

  // state resets:
  // initial pass state disables blending, depth-test, stencil-test, face culling
  // those state pieces must be set/bound every pass
  bindBlendState(disabledBlendState); // disables GL_BLEND
  bindDepthStencilState(disabledDepthStencilState); // disables GL_DEPTH_TEST, GL_STENCIL_TEST
  setCullState(kCullDisabled);
  setDepthBias(0.0f, 0.0f);
}

void RHIGL::setViewport(const RHIRect& viewport) {
  assert(m_activeRenderTarget.get() && "setViewport: can only be called during a render pass.");
  glViewport(viewport.x, viewport.y, viewport.width, viewport.height);
}

void RHIGL::setDepthBias(float slopeScale, float constantBias) {
  assert(m_activeRenderTarget.get() && "setDepthBias: can only be called during a render pass.");
  if (slopeScale == m_currentDepthBiasSlopeScale && constantBias == m_currentDepthBiasConstant)
    return;

  m_currentDepthBiasSlopeScale = slopeScale;
  m_currentDepthBiasConstant = constantBias;

  if (slopeScale == 0.0f && constantBias == 0.0f) {
    glDisable(GL_POLYGON_OFFSET_FILL);
  } else {
    glPolygonOffset(slopeScale, constantBias);
    glEnable(GL_POLYGON_OFFSET_FILL);
  }
}

void RHIGL::endRenderPass(RHIRenderTarget::ptr renderTarget) {
  assert(m_activeRenderTarget.get() && "endRenderPass: beginRenderPass was not previously called.");
  assert(m_activeRenderTarget.get() == static_cast<RHIRenderTargetGL*>(renderTarget.get()) && "endRenderPass: beginRenderPass was called with a different renderTarget.");

  m_activeRenderTarget.reset();
}

void RHIGL::internalPerformBlit(GLuint sourceFBO, bool sourceIsMultisampled, RHIRect destRect, RHIRect sourceRect) {
  if (destRect.empty()) {
    destRect = RHIRect::sized(m_activeRenderTarget->width(), m_activeRenderTarget->height());
  }

  GLenum scaleMode = GL_LINEAR;
  // slight fast-path for same size blits without multisample resolve.
  // (the driver probably detects this anyway)
  if (sourceIsMultisampled && (destRect.width == sourceRect.width) && (destRect.height == sourceRect.height)) {
    scaleMode = GL_NEAREST;
  }

  GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, sourceFBO));
  GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_activeRenderTarget->glFramebufferId()));

  glBlitFramebuffer(
    sourceRect.left(), sourceRect.top(), sourceRect.right(), sourceRect.bottom(),
    destRect.left(), destRect.top(), destRect.right(), destRect.bottom(),
    GL_COLOR_BUFFER_BIT, scaleMode);

#ifndef NDEBUG
  // Work around a bug on Mac OS X that causes a spurious OpenGL error to be emitted after using glBlitFramebuffer
  // to resolve a multisampled framebuffer. (The resolve succeeds, but raises GL_INVALID_OPERATION).
  // We skip the GL() error reporting wrapper on that blit and also clear the error here if GL error reporting
  // is enabled (debug builds only)
  while (glGetError()) {}
#endif

  // reset active framebuffer binding after blit
  GL(glBindFramebuffer(GL_FRAMEBUFFER, m_activeRenderTarget->glFramebufferId()));
}

void RHIGL::blitTex(RHISurface::ptr sourceTexture, uint8_t sourceLayer, RHIRect destRect, RHIRect sourceRect) {
  if (rhi_gl_forceBlitEmulation) {
    blitTex_emulated(m_activeRenderTarget, sourceTexture, sourceLayer, destRect, sourceRect);
    return;
  }

  RHISurfaceGL* glTex = static_cast<RHISurfaceGL*>(sourceTexture.get());
  GLuint sourceFBO = generateSingleLayerFramebufferForTexture(glTex, sourceLayer);

  if (sourceRect.empty())
    sourceRect = RHIRect::sized(glTex->width(), glTex->height());

  internalPerformBlit(sourceFBO, sourceTexture->samples() > 1, destRect, sourceRect);
  glDeleteFramebuffers(1, &sourceFBO);
}
 
void RHIGL::bindRenderPipeline(RHIRenderPipeline::ptr pipeline) {
  bool samePipeline = (m_activeRenderPipeline.get() == static_cast<RHIRenderPipelineGL*>(pipeline.get()));

  // unbind uniform buffers / release ref-ptrs
  for (size_t i = 0; i < 16; ++i) {
    m_activeUniformBuffers[i].reset();
  }

  // debug help: unbind all stream buffers when switching pipelines to make sure that we didn't miss a bind operation
  for (size_t i = 0; i < 16; ++i) {
    m_activeStreamBuffers[i].reset();
  }

  if (samePipeline)
    return; // can skip all the subsequent setup if we're using the same pipeline

  RHIRenderPipelineGL* previousPipeline = m_activeRenderPipeline.get();

  m_activeRenderPipeline = static_cast<RHIRenderPipelineGL*>(pipeline.get());

  // bind shader and VAO
  glUseProgram(m_activeRenderPipeline->shaderGL()->program());
  glBindVertexArray(m_activeRenderPipeline->vao());


  // push pipeline state to GL
  if (m_activeRenderPipeline->descriptor().primitiveTopology == kPrimitiveTopologyPatches) {
    glPatchParameteri(GL_PATCH_VERTICES, m_activeRenderPipeline->descriptor().patchControlPoints);
  }

  if (previousPipeline && previousPipeline->descriptor().rasterizationEnabled == m_activeRenderPipeline->descriptor().rasterizationEnabled) {
    // state unchanged
  } else {
    if (m_activeRenderPipeline->descriptor().rasterizationEnabled) {
      glDisable(GL_RASTERIZER_DISCARD);
    } else {
      glEnable(GL_RASTERIZER_DISCARD);
    }
  }

  if (previousPipeline && previousPipeline->descriptor().alphaToCoverageEnabled == m_activeRenderPipeline->descriptor().alphaToCoverageEnabled) {
    // state unchanged
  } else {
    if (m_activeRenderPipeline->descriptor().alphaToCoverageEnabled) {
      glEnable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    } else {
      glDisable(GL_SAMPLE_ALPHA_TO_COVERAGE);
    }
  }

  if (previousPipeline && previousPipeline->descriptor().perSampleShadingEnabled == m_activeRenderPipeline->descriptor().perSampleShadingEnabled) {
    // state unchanged
  } else {
    if (m_activeRenderPipeline->descriptor().perSampleShadingEnabled) {
      glMinSampleShading(1.0f);
      glEnable(GL_SAMPLE_SHADING);
    } else {
      glDisable(GL_SAMPLE_SHADING);
    }
  }

  if (previousPipeline && previousPipeline->descriptor().primitiveRestartEnabled == m_activeRenderPipeline->descriptor().primitiveRestartEnabled) {
    // state unchanged
  } else {
    if (m_activeRenderPipeline->descriptor().primitiveRestartEnabled) {
      glEnable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    } else {
      glDisable(GL_PRIMITIVE_RESTART_FIXED_INDEX);
    }
  }

}

void RHIGL::bindStreamBuffer(size_t streamIndex, RHIBuffer::ptr buffer) {
  assert(!m_inComputePass && "bindStreamBuffer(): cannot be used during compute pass encoding");
  assert(streamIndex < 16);
  m_activeStreamBuffers[streamIndex] = static_cast<RHIBufferGL*>(buffer.get());
}

static GLenum rhiCompareFunctionToGL(RHIDepthStencilCompareFunction func) {
  switch (func) {
    case kCompareNever: return GL_NEVER;
    case kCompareLess: return GL_LESS;
    case kCompareLessEqual: return GL_LEQUAL;
    case kCompareEqual: return GL_EQUAL;
    case kCompareNotEqual: return GL_NOTEQUAL;
    case kCompareGreaterEqual: return GL_GEQUAL;
    case kCompareGreater: return GL_GREATER;
    case kCompareAlways: return GL_ALWAYS;
    default:
      assert(false && "rhiCompareFunctionToGL: invalid input");
      return GL_NONE;
  };
}

static GLenum rhiStencilOpToGL(RHIStencilOperation op) {
  switch (op) {
    case kStencilKeep: return GL_KEEP;
    case kStencilZero: return GL_ZERO;
    case kStencilReplace: return GL_REPLACE;
    case kStencilIncrementClamp: return GL_INCR;
    case kStencilDecrementClamp: return GL_DECR;
    case kStencilInvert: return GL_INVERT;
    case kStencilIncrementWrap: return GL_INCR_WRAP;
    case kStencilDecrementWrap: return GL_DECR_WRAP;
    default:
      assert(false && "rhiStencilOpToGL: invalid input");
      return GL_NONE;
  };
}

void RHIGL::bindDepthStencilState(RHIDepthStencilState::ptr depthStencilState) {
  assert(inRenderPass() && "bindDepthStencilState: logic error: state bound between passes will be clobbered at the start of the next pass.");

  if (m_activeDepthStencilState == static_cast<RHIDepthStencilStateGL*>(depthStencilState.get()))
    return;

  m_activeDepthStencilState = static_cast<RHIDepthStencilStateGL*>(depthStencilState.get());

  // sync state object to GL
  const RHIDepthStencilStateDescriptor& state = m_activeDepthStencilState->descriptor();
  if (state.depthTestEnable) {
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(rhiCompareFunctionToGL(state.depthFunction));
  } else {
    glDisable(GL_DEPTH_TEST);
  }

  glDepthMask(state.depthWriteEnable ? GL_TRUE : GL_FALSE);

  if (state.stencilTestEnable) {
    glEnable(GL_STENCIL_TEST);
    glStencilFuncSeparate(GL_FRONT, rhiCompareFunctionToGL(state.stencilFront.compareFunc), state.stencilFront.referenceValue, state.stencilMask);
    glStencilFuncSeparate(GL_BACK,  rhiCompareFunctionToGL(state.stencilBack.compareFunc),  state.stencilBack.referenceValue,  state.stencilMask);
    glStencilOpSeparate(GL_FRONT, rhiStencilOpToGL(state.stencilFront.failOp), rhiStencilOpToGL(state.stencilFront.depthFailOp), rhiStencilOpToGL(state.stencilFront.passOp));
    glStencilOpSeparate(GL_BACK,  rhiStencilOpToGL(state.stencilBack.failOp),  rhiStencilOpToGL(state.stencilBack.depthFailOp),  rhiStencilOpToGL(state.stencilBack.passOp));
  } else {
    glDisable(GL_STENCIL_TEST);
  }
}

static GLenum rhiBlendWeightToGL(RHIBlendWeight w) {
  switch (w) {
    case kBlendZero: return GL_ZERO;
    case kBlendOne: return GL_ONE;
    case kBlendSourceColor: return GL_SRC_COLOR;
    case kBlendOneMinusSourceColor: return GL_ONE_MINUS_SRC_COLOR;
    case kBlendDestColor: return GL_DST_COLOR;
    case kBlendOneMinusDestColor: return GL_ONE_MINUS_DST_COLOR;
    case kBlendSourceAlpha: return GL_SRC_ALPHA;
    case kBlendOneMinusSourceAlpha: return GL_ONE_MINUS_SRC_ALPHA;
    case kBlendDestAlpha: return GL_DST_ALPHA;
    case kBlendOneMinusDestAlpha: return GL_ONE_MINUS_DST_ALPHA;
    case kBlendConstantColor: return GL_CONSTANT_COLOR;
    case kBlendOneMinusConstantColor: return GL_ONE_MINUS_CONSTANT_COLOR;
    case kBlendConstantAlpha: return GL_CONSTANT_ALPHA;
    case kBlendOneMinusConstantAlpha: return GL_ONE_MINUS_CONSTANT_ALPHA;
    case kBlendSourceAlphaSaturate: return GL_SRC_ALPHA_SATURATE;
    default:
      assert(false && "rhiBlendWeightToGL: invalid enum value");
      return GL_NONE;
  };
}

static GLenum rhiBlendFuncToGL(RHIBlendFunc f) {
  switch (f) {
    case kBlendFuncAdd: return GL_FUNC_ADD;
    case kBlendFuncSubtract: return GL_FUNC_SUBTRACT;
    case kBlendFuncReverseSubtract: return GL_FUNC_REVERSE_SUBTRACT;
    case kBlendFuncMin: return GL_MIN;
    case kBlendFuncMax: return GL_MAX;
    default:
      assert(false && "rhiBlendFuncToGL: invalid enum value");
      return GL_NONE;
  };
}

void RHIGL::bindBlendState(RHIBlendState::ptr blendState) {
  assert(inRenderPass() && "RHIGL::bindBlendState: logic error: state bound between render passes will be clobbered at the start of the next pass");

  RHIBlendStateGL* glBlendState = static_cast<RHIBlendStateGL*>(blendState.get());
  if (m_activeBlendState.get() == glBlendState)
    return;
  m_activeBlendState = glBlendState;

  const RHIBlendStateDescriptor& desc = glBlendState->descriptor();

  if (desc.targetBlendStates.empty()) {
    glDisable(GL_BLEND);
    return;
  }

  glBlendColor(desc.constantColor[0], desc.constantColor[1], desc.constantColor[2], desc.constantColor[3]);

  if (desc.targetBlendStates.size() == 1) {
    // single global blend state
    const RHIBlendStateDescriptorElement& el = desc.targetBlendStates[0];
    if (!el.blendEnabled) {
      glDisable(GL_BLEND);
      return;
    }

    glEnable(GL_BLEND);
    GL(glBlendFuncSeparate(rhiBlendWeightToGL(el.colorSource), rhiBlendWeightToGL(el.colorDest), rhiBlendWeightToGL(el.alphaSource), rhiBlendWeightToGL(el.alphaDest)));
    GL(glBlendEquationSeparate(rhiBlendFuncToGL(el.colorFunc), rhiBlendFuncToGL(el.alphaFunc)));
  } else {
    // per-target blend states
    for (size_t target = 0; target < desc.targetBlendStates.size(); ++target) {
      const RHIBlendStateDescriptorElement& el = desc.targetBlendStates[target];
      if (!el.blendEnabled) {
        GL(glDisablei(GL_BLEND, target));
      } else {
        GL(glEnablei(GL_BLEND, target));
        GL(glBlendFuncSeparatei(target, rhiBlendWeightToGL(el.colorSource), rhiBlendWeightToGL(el.colorDest), rhiBlendWeightToGL(el.alphaSource), rhiBlendWeightToGL(el.alphaDest)));
        GL(glBlendEquationSeparatei(target, rhiBlendFuncToGL(el.colorFunc), rhiBlendFuncToGL(el.alphaFunc)));
      }
    }
  }

}

void RHIGL::setCullState(RHICullState cullState) {
  if (m_currentCullState == cullState)
    return;
  m_currentCullState = cullState;

  switch (cullState) {
    case kCullDisabled:
      glDisable(GL_CULL_FACE);
      break;

    case kCullFrontFaces:
      glCullFace(GL_FRONT);
      glEnable(GL_CULL_FACE);
      break;

    case kCullBackFaces:
      glCullFace(GL_BACK);
      glEnable(GL_CULL_FACE);
      break;

    default:
      assert(false && "RHIGL::setCullState: invalid enum value");
  }
}

void RHIGL::loadTexture(FxAtomicString name, RHISurface::ptr tex, RHISampler::ptr sampler) {
  int32_t location = m_inComputePass ?
    m_activeComputePipeline->shaderGL()->samplerAttributeLocation(name) :
    m_activeRenderPipeline->shaderGL()->samplerAttributeLocation(name);

  if (location < 0) {
    //printf("RHIGL::loadTexture: no texture slot in current pipeline mapped to name \"%s\"\n", name.c_str());
    return;
  }

  RHISurfaceGL* glTex = static_cast<RHISurfaceGL*>(tex.get());
  RHISamplerGL* glSampler = static_cast<RHISamplerGL*>(sampler.get());

  GL(glActiveTexture(GL_TEXTURE0 + location));
  GL(glBindTexture(glTex->glTarget(), glTex->glId()));
  GL(glBindSampler(location, glSampler ? glSampler->glId() : 0));
}

void RHIGL::loadShaderBuffer(FxAtomicString name, RHIBuffer::ptr buffer) {
  int32_t location = m_inComputePass ?
    m_activeComputePipeline->shaderGL()->bufferBlockLocation(name) :
    m_activeRenderPipeline->shaderGL()->bufferBlockLocation(name);

  if (location < 0) {
    printf("RHIGL::loadShaderBuffer: no buffer slot in current pipeline mapped to name \"%s\"\n", name.c_str());
    return;
  }

  RHIBufferGL* glBuf = static_cast<RHIBufferGL*>(buffer.get());
  GL(glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, glBuf->glId()));
}

void RHIGL::loadUniformBlock(FxAtomicString name, RHIBuffer::ptr blockBuffer) {
  int32_t location = m_inComputePass ?
    m_activeComputePipeline->shaderGL()->uniformBlockLocation(name) :
    m_activeRenderPipeline->shaderGL()->uniformBlockLocation(name);

  if (location < 0) {
    printf("RHIGL::loadUniformBlock: no uniform block slot in current pipeline mapped to name \"%s\"\n", name.c_str());
    return;
  }

  assert(location < 16);

  RHIBufferGL* bufferGL = static_cast<RHIBufferGL*>(blockBuffer.get());
  m_activeUniformBuffers[location] = bufferGL;
  GL(glBindBufferBase(GL_UNIFORM_BUFFER, location, bufferGL->glId()));
}

void RHIGL::loadUniformBlockImmediate(FxAtomicString name, const void* data, size_t size) {
  int32_t location = m_inComputePass ?
    m_activeComputePipeline->shaderGL()->uniformBlockLocation(name) :
    m_activeRenderPipeline->shaderGL()->uniformBlockLocation(name);

  if (location < 0) {
    printf("RHIGL::loadUniformBlockImmediate: no uniform block slot in current pipeline mapped to name \"%s\"\n", name.c_str());
    return;
  }

  size_t alignedDataSize = ((size + (m_uniformBufferOffsetAlignment - 1)) & (~(m_uniformBufferOffsetAlignment - 1)));


  size_t newWriteOffset = m_immediateScratchBufferWriteOffset + alignedDataSize;
  if (newWriteOffset > m_immediateScratchBufferSize) {
    m_immediateScratchBufferWriteOffset = 0;
    newWriteOffset = alignedDataSize;
  }

  memcpy(m_immediateScratchBufferData + m_immediateScratchBufferWriteOffset, data, size);
  GL(glBindBufferRange(GL_UNIFORM_BUFFER, location, m_immediateScratchBufferId, m_immediateScratchBufferWriteOffset, alignedDataSize));
  m_immediateScratchBufferWriteOffset = newWriteOffset;
}

static GLenum convertPrimitiveTopology(RHIPrimitiveTopology topo) {
  switch (topo) {
    case kPrimitiveTopologyPoints: return GL_POINTS;
    case kPrimitiveTopologyLineList: return GL_LINES;
    case kPrimitiveTopologyLineStrip: return GL_LINE_STRIP;
    case kPrimitiveTopologyTriangleList: return GL_TRIANGLES;
    case kPrimitiveTopologyTriangleStrip: return GL_TRIANGLE_STRIP;
    case kPrimitiveTopologyPatches: return GL_PATCHES;
    default: assert(false); return GL_NONE;
  }
}

void RHIGL::internalSetupRenderPipelineState() {
  for (const RHIRenderPipelineGL::StreamBufferDescriptor& bufferDesc : m_activeRenderPipeline->streamBufferDescriptors()) {
    assert(m_activeStreamBuffers[bufferDesc.index] && "RHIGL::internalSetupRenderPipelineState: Missing stream buffer binding");
    glBindVertexBuffer(bufferDesc.index, m_activeStreamBuffers[bufferDesc.index]->glId(), /*offset=*/0, bufferDesc.stride);
  }
}

void RHIGL::drawPrimitives(uint32_t vertexStart, uint32_t vertexCount, uint32_t instanceCount, uint32_t baseInstance) {
#ifndef glDrawArraysInstancedBaseInstanceEXT
#define glDrawArraysInstancedBaseInstanceEXT glDrawArraysInstancedBaseInstance
#endif

  internalSetupRenderPipelineState();
  if (instanceCount > 1) {
    GL(glDrawArraysInstancedBaseInstanceEXT(convertPrimitiveTopology(m_activeRenderPipeline->descriptor().primitiveTopology), vertexStart, vertexCount, instanceCount, baseInstance));
  } else {
    GL(glDrawArrays(convertPrimitiveTopology(m_activeRenderPipeline->descriptor().primitiveTopology), vertexStart, vertexCount));
  }
}

void RHIGL::drawPrimitivesIndirect(RHIBuffer::ptr indirectBuffer, uint32_t indirectCommandCount, uint32_t indirectCommandArrayOffset) {
  internalSetupRenderPipelineState();
  RHIBufferGL* glBuf = static_cast<RHIBufferGL*>(indirectBuffer.get());
  GL(glBindBuffer(GL_DRAW_INDIRECT_BUFFER, glBuf->glId()));

  void* offset = reinterpret_cast<void*>(indirectCommandArrayOffset * 16);
#ifndef glMultiDrawArraysIndirectEXT
#define glMultiDrawArraysIndirectEXT glMultiDrawArraysIndirect
#endif
  if (indirectCommandCount == 1) {
    GL(glDrawArraysIndirect(convertPrimitiveTopology(m_activeRenderPipeline->descriptor().primitiveTopology), offset));
  } else {
    GL(glMultiDrawArraysIndirectEXT(convertPrimitiveTopology(m_activeRenderPipeline->descriptor().primitiveTopology), offset, indirectCommandCount, 0));
  }

  GL(glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0));
}

static GLenum RHIIndexBufferTypeToGL(RHIIndexBufferType indexBufferType) {
  switch (indexBufferType) {
    default:
      assert(false && "RHIIndexBufferTypeToGL: unhandled enum");
    case kIndexBufferTypeUInt16:
      return GL_UNSIGNED_SHORT;
    case kIndexBufferTypeUInt32:
      return GL_UNSIGNED_INT;
  };
}

void RHIGL::drawIndexedPrimitives(RHIBuffer::ptr indexBuffer, RHIIndexBufferType indexBufferType, uint32_t indexCount, uint32_t indexOffsetElements, uint32_t instanceCount, uint32_t baseInstance) {
  internalSetupRenderPipelineState();

  RHIBufferGL* glIndexBuffer = static_cast<RHIBufferGL*>(indexBuffer.get());
  GL(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glIndexBuffer->glId()));

#ifndef glDrawElementsInstancedBaseInstanceEXT
#define glDrawElementsInstancedBaseInstanceEXT glDrawElementsInstancedBaseInstance
#endif

  if (instanceCount > 1) {
    GL(glDrawElementsInstancedBaseInstanceEXT(convertPrimitiveTopology(m_activeRenderPipeline->descriptor().primitiveTopology), indexCount, RHIIndexBufferTypeToGL(indexBufferType), reinterpret_cast<const void*>(indexOffsetElements * RHIIndexBufferTypeSize(indexBufferType)), instanceCount, baseInstance));
  } else {
    GL(glDrawElements(convertPrimitiveTopology(m_activeRenderPipeline->descriptor().primitiveTopology), indexCount, RHIIndexBufferTypeToGL(indexBufferType), reinterpret_cast<const void*>(indexOffsetElements * RHIIndexBufferTypeSize(indexBufferType))));
  }
}

RHITimerQuery::ptr RHIGL::newTimerQuery() {
  return RHITimerQuery::ptr();
}

RHIOcclusionQuery::ptr RHIGL::newOcclusionQuery(RHIOcclusionQueryMode queryMode) {
  return RHIOcclusionQuery::ptr(new RHIOcclusionQueryGL(queryMode));
}

void RHIGL::beginTimerQuery(RHITimerQuery::ptr query) {
  // Not supported in ES3
}

void RHIGL::endTimerQuery(RHITimerQuery::ptr query) {
  // Not supported in ES3
}

void RHIGL::recordTimestamp(RHITimerQuery::ptr query) {
  // Not supported in ES3
}

static GLenum rhiOcclusionQueryModeToGL(RHIOcclusionQueryMode mode) {
  switch (mode) {
    //case kOcclusionQueryModeSampleCount: return GL_SAMPLES_PASSED; // Not supported in ES3
    case kOcclusionQueryModeAnySamplesPassed: return GL_ANY_SAMPLES_PASSED;
    default:
      assert(false && "rhiOcclusionQueryModeToGL: invalid enum value");
      return GL_NONE;
  };
}

void RHIGL::beginOcclusionQuery(RHIOcclusionQuery::ptr query) {
  RHIOcclusionQueryGL* glQuery = static_cast<RHIOcclusionQueryGL*>(query.get());
  GL(glBeginQuery(rhiOcclusionQueryModeToGL(glQuery->mode()), glQuery->glId()));
}

void RHIGL::endOcclusionQuery(RHIOcclusionQuery::ptr query) {
  RHIOcclusionQueryGL* glQuery = static_cast<RHIOcclusionQueryGL*>(query.get());
  GL(glEndQuery(rhiOcclusionQueryModeToGL(glQuery->mode())));
}

uint64_t RHIGL::getQueryResult(RHIOcclusionQuery::ptr query) {
  RHIOcclusionQueryGL* glQuery = static_cast<RHIOcclusionQueryGL*>(query.get());
  GLuint res = 0;
  GL(glGetQueryObjectuiv(glQuery->glId(), GL_QUERY_RESULT, &res));
  return res;
}

uint64_t RHIGL::getQueryResult(RHITimerQuery::ptr query) {
  // Not supported in ES3
  return 0;
}

uint64_t RHIGL::getTimestampImmediate() {
  // Not supported in ES3
  return 0;
}

void RHIGL::pushDebugGroup(const char* groupName) {
#ifdef _WIN32
  if (GLEW_KHR_debug) {
    glPushDebugGroup(GL_DEBUG_SOURCE_APPLICATION, 0, strlen(groupName), groupName);
    return;
  }

  if (!GLEW_EXT_debug_marker)
    return;
#endif

  // Mac OS X always has EXT_debug_marker, that's our fallback.
  glPushGroupMarkerEXT(strlen(groupName), groupName);
}

void RHIGL::popDebugGroup() {
#ifdef _WIN32
  if (GLEW_KHR_debug) {
    glPopDebugGroup();
    return;
  }

  if (!GLEW_EXT_debug_marker)
    return;
#endif

  glPopGroupMarkerEXT();
}

void RHIGL::swapBuffers(RHIRenderTarget::ptr renderTarget) {
  assert(renderTarget->isWindowRenderTarget());
  RHIWindowRenderTargetGL* wrt = static_cast<RHIWindowRenderTargetGL*>(renderTarget.get());
  wrt->platformSwapBuffers();
}

void RHIGL::flush() {
  glFlush();
}

void RHIGL::flushAndWaitForGPUScheduling() {
  // TODO: not sure if this is correct.
  glFlush();
}

uint32_t RHIGL::maxMultisampleSamples() {
  return m_maxMultisampleSamples;
}

bool RHIGL::supportsGeometryShaders() {
  // Minimum feature level is GL4.3, so this will always be true
  return true;
}

void RHIGL::populateGlobalShaderDescriptorEnvironment(RHIShaderDescriptor* descriptor) {
  // allow default impl to populate flags first
  this->RHI::populateGlobalShaderDescriptorEnvironment(descriptor);
  descriptor->setFlag("RHI_GL", true);
  descriptor->setFlag("RHI_METAL", false);
}


void RHIGL::beginComputePass() {
  assert(!m_inComputePass && "beginComputePass: endComputePass() was not called on the prior pass");
  assert(!m_activeRenderTarget && "beginComputePass: currently encoding a render pass");
  m_inComputePass = true;
  // TODO
}

void RHIGL::endComputePass() {
  assert(m_inComputePass && "endComputePass: beginComputePass() was not called");
  m_activeComputePipeline.reset();
  m_inComputePass = false;
  // TODO

  // Ensure that writes initiated by this pass are finished before the next pass starts
  glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
}

void RHIGL::bindComputePipeline(RHIComputePipeline::ptr pipeline) {
  assert(m_inComputePass && "bindComputePipeline: not encoding a compute pass");
  m_activeComputePipeline = static_cast<RHIComputePipelineGL*>(pipeline.get());
  GL(glUseProgram(m_activeComputePipeline->shaderGL()->program()));
}

void RHIGL::dispatchCompute(uint32_t threadgroupCountX, uint32_t threadgroupCountY, uint32_t threadgroupCountZ) {
  assert(m_inComputePass);
  assert(m_activeComputePipeline);

  GL(glDispatchCompute(threadgroupCountX, threadgroupCountY, threadgroupCountZ));
}

void RHIGL::dispatchComputeIndirect(RHIBuffer::ptr buffer) {
  assert(m_inComputePass);
  assert(m_activeComputePipeline);

  RHIBufferGL* glBuf = static_cast<RHIBufferGL*>(buffer.get());
  GL(glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, glBuf->glId()));
  GL(glDispatchComputeIndirect(NULL));
  GL(glBindBuffer(GL_DISPATCH_INDIRECT_BUFFER, 0));
}

