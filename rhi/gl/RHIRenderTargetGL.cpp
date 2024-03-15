#include "rhi/gl/RHIRenderTargetGL.h"
#include "rhi/gl/RHISurfaceGL.h"

RHIRenderTargetGL::RHIRenderTargetGL() : m_glFramebufferId(0), m_width(0), m_height(0), m_layers(1), m_samples(1), m_colorTargetCount(0), m_isArray(false), m_hasDepthStencilTarget(false) {

}

RHIRenderTargetGL::~RHIRenderTargetGL() {
  if (m_glFramebufferId)
    glDeleteFramebuffers(1, &m_glFramebufferId);

}

/*static*/ bool RHIRenderTargetGL::formatCheck(RHIRenderTargetGL* target, const RHIRenderTargetDescriptorElement& element, bool& haveFormat) {
  if (!haveFormat) {
    target->m_width = element.surface->width();
    target->m_height = element.surface->height();
    target->m_layers = element.singleLayer ? 1 : element.surface->layers();
    target->m_samples = element.surface->samples();
    target->m_isArray = element.singleLayer ? false : element.surface->isArray();

    haveFormat = true;
  } else if (target->m_width != element.surface->width() ||
             target->m_height != element.surface->height() ||
             target->m_layers != (element.singleLayer ? 1 : element.surface->layers()) ||
             target->m_samples != element.surface->samples() ||
             target->m_isArray != (element.singleLayer ? false : element.surface->isArray())) {

    assert(false && "RHIRenderTargetGL: Format mismatch in supplied color targets");
    return false;
  }
  return true;
}

/*static*/ void RHIRenderTargetGL::handleAttachment(GLenum attachPoint, const RHIRenderTargetDescriptorElement& element) {
  RHISurfaceGL* glSurface = static_cast<RHISurfaceGL*>(element.surface.get());
  if (glSurface->isGLTexture()) {
    if (element.singleLayer) {
      GL(glFramebufferTextureLayer(GL_FRAMEBUFFER, attachPoint, glSurface->glId(), element.level, element.layerIndex));
    } else {
      GL(glFramebufferTexture(GL_FRAMEBUFFER, attachPoint, glSurface->glId(), element.level));
    }
  } else {
    assert(glSurface->isGLRenderbuffer());
    GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, attachPoint, GL_RENDERBUFFER, glSurface->glId()));
  }
}

/*static*/ RHIRenderTargetGL* RHIRenderTargetGL::newRenderTarget(const RHIRenderTargetDescriptor& descriptor) {
  assert(descriptor.colorTargets.size() || descriptor.depthStencilTarget.surface);

  RHIRenderTargetGL* tgt = new RHIRenderTargetGL();
  glGenFramebuffers(1, &tgt->m_glFramebufferId);
  glBindFramebuffer(GL_FRAMEBUFFER, tgt->m_glFramebufferId);

  tgt->m_colorTargetCount = descriptor.colorTargets.size();
  tgt->m_hasDepthStencilTarget = (descriptor.depthStencilTarget.surface.get() != NULL);

  // enforce size and format consistency
  bool haveFormat = false;

  for (size_t i = 0; i < descriptor.colorTargets.size(); ++i) {
    formatCheck(tgt, descriptor.colorTargets[i], haveFormat);
    handleAttachment(GL_COLOR_ATTACHMENT0 + i, descriptor.colorTargets[i]);
  }
 
  if (descriptor.depthStencilTarget.surface) {
    formatCheck(tgt, descriptor.depthStencilTarget, haveFormat);

    GLenum attachPoint = GL_NONE;

    switch (descriptor.depthStencilTarget.surface->format()) {
      case kSurfaceFormat_Depth16:
      case kSurfaceFormat_Depth32f:
        attachPoint = GL_DEPTH_ATTACHMENT;
        break;
      case kSurfaceFormat_Depth32f_Stencil8:
        attachPoint = GL_DEPTH_STENCIL_ATTACHMENT;
        break;
      case kSurfaceFormat_Stencil8:
        attachPoint = GL_STENCIL_ATTACHMENT;
        break;

      default:
        assert(false && "RHIRenderTargetGL: unknown RHI format for depth-stencil target");
    }

    handleAttachment(attachPoint, descriptor.depthStencilTarget);
  }
  
  GLenum framebufferStatus = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (framebufferStatus != GL_FRAMEBUFFER_COMPLETE) {
    fprintf(stderr, "RHIRenderTargetGL: Framebuffer setup error, status is 0x%x\n", framebufferStatus);
    assert(false && "RHIRenderTargetGL: Framebuffer setup failed");
  }

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

  // Setup drawBuffers state (which belongs to the FBO)
  // this just picks the first (m_colorBuffersCount) buffers from the array which contains all of the COLOR_ATTACHMENT enums in order.
  if (tgt->m_colorTargetCount > 0) {
    GL(glDrawBuffers(tgt->m_colorTargetCount, buffers));
  }

  return tgt;
}

uint32_t RHIRenderTargetGL::width() const {
  return m_width;
}
uint32_t RHIRenderTargetGL::height() const {
  return m_height;
}
uint32_t RHIRenderTargetGL::layers() const {
  return m_layers;
}
uint32_t RHIRenderTargetGL::samples() const {
  return m_samples;
}
bool RHIRenderTargetGL::isArray() const {
  return m_isArray;
}
bool RHIRenderTargetGL::isWindowRenderTarget() const {
  return false;
}
size_t RHIRenderTargetGL::colorTargetCount() const {
  return m_colorTargetCount;
}
bool RHIRenderTargetGL::hasDepthStencilTarget() const {
  return m_hasDepthStencilTarget;
}
