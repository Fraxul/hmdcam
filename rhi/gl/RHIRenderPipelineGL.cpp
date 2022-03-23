#include "rhi/gl/RHIRenderPipelineGL.h"

RHIRenderPipelineGL::RHIRenderPipelineGL(RHIShaderGL::ptr shader, const RHIRenderPipelineDescriptor& descriptor) : m_shader(shader), m_descriptor(descriptor), m_vao(0) {
  assert(shader);

}

GLuint RHIRenderPipelineGL::vao() {
  // TODO VAOs aren't shared between contexts, need per-RHI-instance cache
  if (m_vao)
    return m_vao;

  glGenVertexArrays(1, &m_vao);
  glBindVertexArray(m_vao);

  // setup VAO format
  const RHIVertexLayout& vertexLayout = shaderGL()->vertexLayout();
  for (size_t i = 0; i < vertexLayout.elements.size(); ++i) {
    const RHIVertexLayoutElement& elem = vertexLayout.elements[i];
    GLint loc = shaderGL()->varyingAttributeLocation(elem.elementName);
    if (loc < 0)
      continue; // fine
    if (elem.elementType == kVertexElementTypeNone)
      continue; // ?

    // find-and-verify or create stream buffer descriptor for this element
    {
      bool haveStreamBufferDescriptor = false;
      for (const RHIRenderPipelineGL::StreamBufferDescriptor& bufferDesc : m_streamBufferDescriptors) {
        if (bufferDesc.index != elem.streamBufferIndex)
          continue;
        haveStreamBufferDescriptor = true;
        assert(bufferDesc.stride == elem.stride);
        assert(bufferDesc.elementFrequency == elem.elementFrequency);
      }
      if (!haveStreamBufferDescriptor) {
        RHIRenderPipelineGL::StreamBufferDescriptor bufferDesc;
        bufferDesc.index = elem.streamBufferIndex;
        bufferDesc.stride = elem.stride;
        bufferDesc.elementFrequency = elem.elementFrequency;
        m_streamBufferDescriptors.push_back(bufferDesc);

        GL(glVertexBindingDivisor(elem.streamBufferIndex, (elem.elementFrequency == kVertexElementFrequencyVertex ? 0 : 1)));
      }
    }

    for (size_t arrayIndex = 0; arrayIndex < elem.arrayElementCount; ++arrayIndex) {
      switch (elem.elementType) {
        case kVertexElementTypeFloat1:
          glVertexAttribFormat(loc + arrayIndex, 1, GL_FLOAT, GL_FALSE, elem.offset + (arrayIndex * 4)); break;
        case kVertexElementTypeFloat2:
          glVertexAttribFormat(loc + arrayIndex, 2, GL_FLOAT, GL_FALSE, elem.offset + (arrayIndex * 8)); break;
        case kVertexElementTypeFloat3:
          glVertexAttribFormat(loc + arrayIndex, 3, GL_FLOAT, GL_FALSE, elem.offset + (arrayIndex * 12)); break;
        case kVertexElementTypeFloat4:
          glVertexAttribFormat(loc + arrayIndex, 4, GL_FLOAT, GL_FALSE, elem.offset + (arrayIndex * 16)); break;

        case kVertexElementTypeHalf1:
          glVertexAttribFormat(loc + arrayIndex, 1, GL_HALF_FLOAT, GL_FALSE, elem.offset + (arrayIndex * 2)); break;
        case kVertexElementTypeHalf2:
          glVertexAttribFormat(loc + arrayIndex, 2, GL_HALF_FLOAT, GL_FALSE, elem.offset + (arrayIndex * 4)); break;
        case kVertexElementTypeHalf4:
          glVertexAttribFormat(loc + arrayIndex, 4, GL_HALF_FLOAT, GL_FALSE, elem.offset + (arrayIndex * 8)); break;

        case kVertexElementTypeInt1:
          glVertexAttribIFormat(loc + arrayIndex, 1, GL_INT, elem.offset + (arrayIndex * 4)); break;
        case kVertexElementTypeInt2:
          glVertexAttribIFormat(loc + arrayIndex, 2, GL_INT, elem.offset + (arrayIndex * 8)); break;
        case kVertexElementTypeInt3:
          glVertexAttribIFormat(loc + arrayIndex, 3, GL_INT, elem.offset + (arrayIndex * 12)); break;
        case kVertexElementTypeInt4:
          glVertexAttribIFormat(loc + arrayIndex, 4, GL_INT, elem.offset + (arrayIndex * 16)); break;

        case kVertexElementTypeUInt1:
          glVertexAttribIFormat(loc + arrayIndex, 1, GL_UNSIGNED_INT, elem.offset + (arrayIndex * 4)); break;
        case kVertexElementTypeUInt2:
          glVertexAttribIFormat(loc + arrayIndex, 2, GL_UNSIGNED_INT, elem.offset + (arrayIndex * 8)); break;
        case kVertexElementTypeUInt3:
          glVertexAttribIFormat(loc + arrayIndex, 3, GL_UNSIGNED_INT, elem.offset + (arrayIndex * 12)); break;
        case kVertexElementTypeUInt4:
          glVertexAttribIFormat(loc + arrayIndex, 4, GL_UNSIGNED_INT, elem.offset + (arrayIndex * 16)); break;

        case kVertexElementTypeUByte1:
          glVertexAttribIFormat(loc + arrayIndex, 1, GL_UNSIGNED_BYTE, elem.offset + (arrayIndex * 1)); break;
        case kVertexElementTypeUByte2:
          glVertexAttribIFormat(loc + arrayIndex, 2, GL_UNSIGNED_BYTE, elem.offset + (arrayIndex * 2)); break;
        case kVertexElementTypeUByte3:
          glVertexAttribIFormat(loc + arrayIndex, 3, GL_UNSIGNED_BYTE, elem.offset + (arrayIndex * 3)); break;
        case kVertexElementTypeUByte4:
          glVertexAttribIFormat(loc + arrayIndex, 4, GL_UNSIGNED_BYTE, elem.offset + (arrayIndex * 4)); break;

        case kVertexElementTypeUByte1N:
          glVertexAttribFormat(loc + arrayIndex, 1, GL_UNSIGNED_BYTE, GL_TRUE, elem.offset + (arrayIndex * 1)); break;
        case kVertexElementTypeUByte2N:
          glVertexAttribFormat(loc + arrayIndex, 2, GL_UNSIGNED_BYTE, GL_TRUE, elem.offset + (arrayIndex * 2)); break;
        case kVertexElementTypeUByte3N:
          glVertexAttribFormat(loc + arrayIndex, 3, GL_UNSIGNED_BYTE, GL_TRUE, elem.offset + (arrayIndex * 3)); break;
        case kVertexElementTypeUByte4N:
          glVertexAttribFormat(loc + arrayIndex, 4, GL_UNSIGNED_BYTE, GL_TRUE, elem.offset + (arrayIndex * 4)); break;

        case kVertexElementTypeByte1N:
          glVertexAttribFormat(loc + arrayIndex, 1, GL_BYTE, GL_TRUE, elem.offset + (arrayIndex * 1)); break;
        case kVertexElementTypeByte2N:
          glVertexAttribFormat(loc + arrayIndex, 2, GL_BYTE, GL_TRUE, elem.offset + (arrayIndex * 2)); break;
        case kVertexElementTypeByte3N:
          glVertexAttribFormat(loc + arrayIndex, 3, GL_BYTE, GL_TRUE, elem.offset + (arrayIndex * 3)); break;
        case kVertexElementTypeByte4N:
          glVertexAttribFormat(loc + arrayIndex, 4, GL_BYTE, GL_TRUE, elem.offset + (arrayIndex * 4)); break;

        case kVertexElementTypeUShort1:
          glVertexAttribIFormat(loc + arrayIndex, 1, GL_UNSIGNED_SHORT, elem.offset + (arrayIndex * 2)); break;
        case kVertexElementTypeUShort2:
          glVertexAttribIFormat(loc + arrayIndex, 2, GL_UNSIGNED_SHORT, elem.offset + (arrayIndex * 4)); break;
        case kVertexElementTypeUShort4:
          glVertexAttribIFormat(loc + arrayIndex, 4, GL_UNSIGNED_SHORT, elem.offset + (arrayIndex * 8)); break;

        case kVertexElementTypeUShort1N:
          glVertexAttribFormat(loc + arrayIndex, 1, GL_UNSIGNED_SHORT, GL_TRUE, elem.offset + (arrayIndex * 2)); break;
        case kVertexElementTypeUShort2N:
          glVertexAttribFormat(loc + arrayIndex, 2, GL_UNSIGNED_SHORT, GL_TRUE, elem.offset + (arrayIndex * 4)); break;
        case kVertexElementTypeUShort4N:
          glVertexAttribFormat(loc + arrayIndex, 4, GL_UNSIGNED_SHORT, GL_TRUE, elem.offset + (arrayIndex * 8)); break;

        case kVertexElementTypeShort1N:
          glVertexAttribFormat(loc + arrayIndex, 1, GL_SHORT, GL_TRUE, elem.offset + (arrayIndex * 2)); break;
        case kVertexElementTypeShort2N:
          glVertexAttribFormat(loc + arrayIndex, 2, GL_SHORT, GL_TRUE, elem.offset + (arrayIndex * 4)); break;
        case kVertexElementTypeShort4N:
          glVertexAttribFormat(loc + arrayIndex, 4, GL_SHORT, GL_TRUE, elem.offset + (arrayIndex * 8)); break;

        default:
          assert(false && "unhandled RHIVertexElementType");
      };
      GL(glVertexAttribBinding(loc + arrayIndex, elem.streamBufferIndex));
      GL(glEnableVertexAttribArray(loc + arrayIndex));
    }
  }

  return m_vao;
}

RHIRenderPipelineGL::~RHIRenderPipelineGL() {
  if (m_vao)
    glDeleteVertexArrays(1, &m_vao);
}

RHIShader* RHIRenderPipelineGL::shader() const {
  return m_shader.get();
}


