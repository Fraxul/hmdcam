#include "rhi/gl/RHIShaderGL.h"
#include <boost/scoped_array.hpp>
#include <set>

static /*CVar*/ bool shader_debug = false;

static GLenum glEnumForShadingUnit(RHIShaderDescriptor::ShadingUnit unit) {
  switch (unit) {
    case RHIShaderDescriptor::kVertexShader:
      return GL_VERTEX_SHADER;
    case RHIShaderDescriptor::kGeometryShader:
      return GL_GEOMETRY_SHADER;
    case RHIShaderDescriptor::kFragmentShader:
      return GL_FRAGMENT_SHADER;
    case RHIShaderDescriptor::kTessEvaluationShader:
      return GL_TESS_EVALUATION_SHADER;
    case RHIShaderDescriptor::kTessControlShader:
      return GL_TESS_CONTROL_SHADER;
    case RHIShaderDescriptor::kComputeShader:
      return GL_COMPUTE_SHADER;

    default:
      assert(false && "glEnumForShadingUnit: bad input");
      return 0;
  };
}

GLuint RHIShaderGL::compileShader(RHIShaderDescriptor::ShadingUnit type, const char* source) {
  GLuint shader = GL(glCreateShader(glEnumForShadingUnit(type)));
  if (!shader) return 0;
  
  GL(glShaderSource(shader, 1, &source, NULL));
  GL(glCompileShader(shader));

  GLint infoLen = 0;
  GL(glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen));
  if (infoLen) {
    char* buf = new char[infoLen];
    GL(glGetShaderInfoLog(shader, infoLen, NULL, buf));
    fprintf(stderr, "Shader info log: %s\n", buf);
    delete[] buf;
  }
  
  GLint compiled = 0;
  GL(glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled));
  if (!compiled) {
    fprintf(stderr, "Could not compile %s shader; see preceeding log. Source follows:\n\n-----\n", RHIShaderDescriptor::nameForShadingUnit(static_cast<RHIShaderDescriptor::ShadingUnit>(type)));
    RHIShaderDescriptor::dumpFormattedShaderSource(source);
    fprintf(stderr, "-----\n\n");
    m_descriptor.debugDumpSourceMap();
    GL(glDeleteShader(shader));

    assert(false && "Shader compilation failed");
    shader = 0;
  }
  return shader;
}

static void dumpShaderInfoLog(GLuint program) {
  GLint len = 0;
  GL(glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len));
  if (len) {
    char* buf = new char[len];
    GL(glGetProgramInfoLog(program, len, NULL, buf));
    fprintf(stderr, "Shader info: %s\n", buf);
    delete[] buf;
  }
}

void RHIShaderGL::internalHandleLinkFailure() {

  m_descriptor.debugDumpSourceMap();

  std::map<RHIShaderDescriptor::ShadingUnit, std::string> unitSources = m_descriptor.preprocessSource();

  fprintf(stderr, "Dumping source of all inputs\n");
  for (std::map<RHIShaderDescriptor::ShadingUnit, std::string>::iterator unit_it = unitSources.begin(); unit_it != unitSources.end(); ++unit_it) {
    fprintf(stderr, " --- %s Source --- \n", RHIShaderDescriptor::nameForShadingUnit(unit_it->first));
    RHIShaderDescriptor::dumpFormattedShaderSource(unit_it->second.c_str());
  }

  fprintf(stderr, "Shader link failed.\n");
  dumpShaderInfoLog(m_program);

  GL(glDeleteProgram(m_program));
  m_program = 0;

  assert(false && "Shader link failure");
}

RHIShaderGL::RHIShaderGL(const RHIShaderDescriptor& descriptor) : m_program(0), m_descriptor(descriptor), m_vertexLayout(descriptor.vertexLayout()) {
  m_program = GL(glCreateProgram());

  std::map<RHIShaderDescriptor::ShadingUnit, std::string> unitSources = m_descriptor.preprocessSource();

  // Shader compilation per unit type
  for (std::map<RHIShaderDescriptor::ShadingUnit, std::string>::iterator unit_it = unitSources.begin(); unit_it != unitSources.end(); ++unit_it) {
    GL(glAttachShader(m_program, compileShader(unit_it->first, unit_it->second.c_str())));
  }

  GL(glLinkProgram(m_program));

  GLint linkStatus = GL_FALSE;

  GL(glGetProgramiv(m_program, GL_LINK_STATUS, &linkStatus));
  if (linkStatus != GL_TRUE) {
    internalHandleLinkFailure();
  }

  // (Re)Populate the uniform/varying attribute arrays
  {
    GLint uniformCount = 0;
    GL(glGetProgramInterfaceiv(m_program, GL_UNIFORM, GL_ACTIVE_RESOURCES, &uniformCount));

    GLsizei bufferSize = 1024;
    boost::scoped_array<char> buffer(new char[bufferSize]);

    for (GLint idx = 0; idx < uniformCount; ++idx) {
      GLsizei nameLength = 0;

      glGetProgramResourceName(m_program, GL_UNIFORM, idx, bufferSize, &nameLength, buffer.get());

      Attribute attr;
      attr.name = FxAtomicString(buffer.get());
      {
        GLenum resourceType = GL_TYPE;
        GLenum resourceLocation = GL_LOCATION;
        glGetProgramResourceiv(m_program, GL_UNIFORM, idx, 1, &resourceType, 1, NULL, (GLint*) &attr.type);
        glGetProgramResourceiv(m_program, GL_UNIFORM, idx, 1, &resourceLocation, 1, NULL, &attr.location);
      }


      switch (attr.type) {
        // separate out sampler attributes from other types of uniforms
        case GL_SAMPLER_BUFFER:
        case GL_INT_SAMPLER_BUFFER:
        case GL_UNSIGNED_INT_SAMPLER_BUFFER:
        case GL_IMAGE_BUFFER:
        case GL_INT_IMAGE_BUFFER:
        case GL_UNSIGNED_INT_IMAGE_BUFFER:
          printf("RHIShaderGL: Warning: Program accepts a buffer sampler parameter \"%s\" which is no longer supported.\n", buffer.get());

        case GL_SAMPLER_2D:
        case GL_SAMPLER_3D:
        case GL_SAMPLER_CUBE:
        case GL_SAMPLER_2D_SHADOW:
        case GL_SAMPLER_2D_ARRAY:
        case GL_SAMPLER_CUBE_MAP_ARRAY:
        case GL_SAMPLER_2D_ARRAY_SHADOW:
        case GL_SAMPLER_2D_MULTISAMPLE:
        case GL_SAMPLER_2D_MULTISAMPLE_ARRAY:
        case GL_SAMPLER_CUBE_SHADOW:
        case GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW:
        case GL_INT_SAMPLER_2D:
        case GL_INT_SAMPLER_3D:
        case GL_INT_SAMPLER_CUBE:
        case GL_INT_SAMPLER_CUBE_MAP_ARRAY:
        case GL_INT_SAMPLER_2D_MULTISAMPLE:
#ifdef GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY_OES
        case GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY_OES:
#endif
        case GL_UNSIGNED_INT_SAMPLER_2D:
        case GL_UNSIGNED_INT_SAMPLER_3D:
        case GL_UNSIGNED_INT_SAMPLER_CUBE:
        case GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY:
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE:
#ifdef GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY_OES
        case GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY_OES:
#endif
#ifdef GL_SAMPLER_EXTERNAL_OES
        case GL_SAMPLER_EXTERNAL_OES:
#endif
#ifdef GL_SAMPLER_EXTERNAL_2D_Y2Y_EXT
        case GL_SAMPLER_EXTERNAL_2D_Y2Y_EXT:
#endif

          if (shader_debug) {
            printf("sampler [%d]: \"%s\"; type %x\n", attr.location, buffer.get(), attr.type);
          }

          m_samplerAttributes.push_back(attr);
          break;

        // Separate out image units from texture samplers, since they're a different
        // resource type and binding namespace
        case GL_IMAGE_2D:
        case GL_IMAGE_3D:
        case GL_IMAGE_CUBE:
        case GL_IMAGE_2D_ARRAY:
        case GL_IMAGE_CUBE_MAP_ARRAY:
        case GL_INT_IMAGE_2D:
        case GL_INT_IMAGE_3D:
        case GL_INT_IMAGE_CUBE:
        case GL_INT_IMAGE_2D_ARRAY:
        case GL_INT_IMAGE_CUBE_MAP_ARRAY:
        case GL_UNSIGNED_INT_IMAGE_2D:
        case GL_UNSIGNED_INT_IMAGE_3D:
        case GL_UNSIGNED_INT_IMAGE_CUBE:
        case GL_UNSIGNED_INT_IMAGE_2D_ARRAY:
        case GL_UNSIGNED_INT_IMAGE_CUBE_MAP_ARRAY:

          if (shader_debug) {
            printf("image [%d]: \"%s\"; type %x\n", attr.location, buffer.get(), attr.type);
          }

          m_imageAttributes.push_back(attr);
          break;


        default:
          if (shader_debug) {
            printf("uniform [%d]: \"%s\"; type %x\n", attr.location, buffer.get(), attr.type);
          }

          // fail if the uniform is in the default block. (uniforms in a non-default block report a location of -1)
          // (this can also indicate an incorrectly identified sampler attribute)
          if (attr.location >= 0) {
            fprintf(stderr, "RHIShaderGL: FATAL: Program accepts free uniform parameter \"%s\" which is no longer supported. All uniforms need to be in a uniform block.\n", buffer.get());
            internalHandleLinkFailure();
          }

          break;
      };

    }

  }

  // Uniform blocks
  {
    m_uniformBlocks.clear();
    GLint uniformBlockCount = 0;
    GL(glGetProgramiv(m_program, GL_ACTIVE_UNIFORM_BLOCKS, &uniformBlockCount));

    GLsizei bufferSize = 0;
    GL(glGetProgramiv(m_program, GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH, &bufferSize));
    ++bufferSize;
    boost::scoped_array<char> buffer(new char[bufferSize]);

    for (GLint idx = 0; idx < uniformBlockCount; ++idx) {
      GLsizei nameLength = 0;
      GLint dataSize = 0;
      GL(glGetActiveUniformBlockName(m_program, idx, bufferSize, &nameLength, buffer.get()));
      GL(glGetActiveUniformBlockiv(m_program, idx, GL_UNIFORM_BLOCK_DATA_SIZE, &dataSize));

      Attribute attr;
      attr.name = FxAtomicString(buffer.get());
      attr.type = 0;
      attr.location = idx;
      if (shader_debug) {
        printf("uniform block [%d]: (%u bytes) \"%s\"\n", attr.location, dataSize, buffer.get());
      }

      m_uniformBlocks.push_back(attr);
    }
  }

  // Buffer blocks
  {
    m_bufferBlocks.clear();
    int bufferCount = 0;
    int nameBufferSize = 0;
    boost::scoped_array<char> nameBuffer;
    GL(glGetProgramInterfaceiv(m_program, GL_SHADER_STORAGE_BLOCK, GL_ACTIVE_RESOURCES, &bufferCount));
    GL(glGetProgramInterfaceiv(m_program, GL_SHADER_STORAGE_BLOCK, GL_MAX_NAME_LENGTH, &nameBufferSize));
    if (bufferCount && nameBufferSize) {
      nameBuffer.reset(new char[nameBufferSize]);
    }
    for (int bufferIdx = 0; bufferIdx < bufferCount; ++bufferIdx) {
      GLsizei nameLength = 0;
      glGetProgramResourceName(m_program, GL_SHADER_STORAGE_BLOCK, bufferIdx, nameBufferSize, &nameLength, nameBuffer.get());

      Attribute attr;
      attr.name = FxAtomicString(nameBuffer.get());
      attr.type = GL_SHADER_STORAGE_BLOCK;
      attr.location = bufferIdx;
      attr.textureUnit = bufferIdx;

      if (shader_debug) {
        printf("SSBO block [%d]: \"%s\"\n", bufferIdx, attr.name.c_str());
      }

      m_bufferBlocks.push_back(attr);
    }
  }

  // Varying attributes
  {
    m_varyingAttributes.clear();
    GLint varyingCount = 0;
    GL(glGetProgramiv(m_program, GL_ACTIVE_ATTRIBUTES, &varyingCount));

    GLsizei bufferSize = 0;
    glGetProgramiv(m_program, GL_ACTIVE_ATTRIBUTE_MAX_LENGTH, &bufferSize);
    ++bufferSize;
    boost::scoped_array<char> buffer(new char[bufferSize]);

    for (GLint idx = 0; idx < varyingCount; ++idx) {
      GLsizei nameLength = 0;
      GLint attributeSize = 0;
      GLenum attributeType = 0;

      GL(glGetActiveAttrib(m_program, idx, bufferSize, &nameLength, &attributeSize, &attributeType, buffer.get()));

      // Workaround for inconsistent naming of array attributes between different drivers.
      // Some drivers append a [0] on the end of an array attribute name
      if ((buffer[nameLength - 1] == ']') && (buffer[nameLength - 2] == '0') && (buffer[nameLength - 3] == '[')) {
        buffer[nameLength - 3] = '\0';
      }

      Attribute attr;
      attr.name = FxAtomicString(buffer.get());
      attr.type = attributeType;
      attr.location = glGetAttribLocation(m_program, attr.name);

      if (shader_debug) {
        printf("varying [%d]: \"%s\"; type %x (size=%d) \n", attr.location, buffer.get(), attributeType, attributeSize);
      }

      // Sanity check: make sure that all varying attributes are represented in the vertex layout.
      {
        bool attributePresentInVertexLayout = false;
        for (size_t i = 0; i < m_vertexLayout.elements.size(); ++i) {
          if (m_vertexLayout.elements[i].elementName == attr.name) {
            attributePresentInVertexLayout = true;
            break;
          }
        }
        if (!attributePresentInVertexLayout) {
          // ignore spurious errors on builtin variables. NVidia drivers would otherwise cause us
          // to complain about a missing binding for variables like gl_VertexID.
          if (memcmp(attr.name.c_str(), "gl_", 3) == 0)
            continue;

          printf("RHIShaderGL: ERROR: Supplied vertex layout does provide a location for varying input \"%s\"\n", attr.name.c_str());
          assert(false && "Shader compiler sanity check failed: vertex layout mismatch (missing input)");
        }
      }

      m_varyingAttributes.push_back(attr);
    }
  }

  // Assign texture unit bindings (identity)
  for (size_t idx = 0; idx < m_samplerAttributes.size(); ++idx) {
    GLint loc = m_samplerAttributes[idx].location;
    if (shader_debug) {
      printf("sampler [loc %d -> unit %zu]: \"%s\"; type %x\n", loc, idx, m_samplerAttributes[idx].name.c_str(), m_samplerAttributes[idx].type);
    }
    GL(glProgramUniform1i(m_program, loc, idx));
    m_samplerAttributes[idx].textureUnit = idx;
  }

  // Assign image unit bindings
  {
    std::set<GLint> seenBindings;
    for (size_t idx = 0; idx < m_imageAttributes.size(); ++idx) {
      GLint loc = m_imageAttributes[idx].location;

      GLint binding = -1;
      GL(glGetUniformiv(m_program, loc, &binding));
      if (seenBindings.find(binding) != seenBindings.end()) {
        printf("RHIShaderGL: ERROR: Image unit bindings are incomplete -- duplicate binding index %d for variable %s\n", binding, m_imageAttributes[idx].name.c_str());
        printf("All Image unit bindings must be specified with a layout(binding=...) construct.\n");
        assert(false && "Shader linking failed: image unit bindings are incomplete.");
      }

      if (shader_debug) {
        printf("image [loc %d -> unit %d]: \"%s\"; type %x\n", loc, binding, m_imageAttributes[idx].name.c_str(), m_imageAttributes[idx].type);
      }
      m_imageAttributes[idx].textureUnit = binding;
    }
  }

  // Assign uniform block bindings (identity)
  for (size_t idx = 0; idx < m_uniformBlocks.size(); ++idx) {
    GL(glUniformBlockBinding(m_program, m_uniformBlocks[idx].location, m_uniformBlocks[idx].location));
  }

  // Query compute shader workgroup size
  if (unitSources.find(RHIShaderDescriptor::kComputeShader) != unitSources.end()) {
    GLint workgroupSize[3];
    GL(glGetProgramiv(m_program, GL_COMPUTE_WORK_GROUP_SIZE, workgroupSize));
    for (size_t i = 0; i < 3; ++i)
      m_localWorkgroupSize[i] = workgroupSize[i];
  }

}

RHIShaderGL::~RHIShaderGL() {
  if (m_program)
    glDeleteProgram(m_program);
}

// The attribute dictionaries are so small and the atomic string comparison is so trivial
// that a simple linear scan of a vector<> here will actually be cheaper than using a map<>
int32_t RHIShaderGL::uniformBlockLocation(const FxAtomicString& name) {
  for (size_t i = 0; i < m_uniformBlocks.size(); ++i) {
    if (m_uniformBlocks[i].name == name)
      return m_uniformBlocks[i].location;
  }
  // printf("[Shader %p] no uniform block attribute named %s\n", this, static_cast<const char*>(name));
  return -1;
}

int32_t RHIShaderGL::varyingAttributeLocation(const FxAtomicString& name) {
  for (size_t i = 0; i < m_varyingAttributes.size(); ++i) {
    if (m_varyingAttributes[i].name == name)
      return m_varyingAttributes[i].location;
  }
  // printf("no varying attribute named %s\n", static_cast<const char*>(name));
  return -1;
}

bool RHIShaderGL::samplerAttributeBinding(const FxAtomicString& name, uint32_t& outLocation, uint32_t& outTextureUnitNumber) {
  for (size_t i = 0; i < m_samplerAttributes.size(); ++i) {
    if (m_samplerAttributes[i].name == name) {
      outLocation = m_samplerAttributes[i].location;
      outTextureUnitNumber =  m_samplerAttributes[i].textureUnit;
      return true;
    }
  }
  return false;
}

bool RHIShaderGL::imageAttributeBinding(const FxAtomicString& name, uint32_t& outLocation, uint32_t& outImageUnitNumber) {
  for (size_t i = 0; i < m_imageAttributes.size(); ++i) {
    if (m_imageAttributes[i].name == name) {
      outLocation = m_imageAttributes[i].location;
      outImageUnitNumber =  m_imageAttributes[i].textureUnit;
      return true;
    }
  }
  return false;
}

int32_t RHIShaderGL::bufferBlockLocation(const FxAtomicString& name) {
  for (size_t i = 0; i < m_bufferBlocks.size(); ++i) {
    if (m_bufferBlocks[i].name == name)
      return m_bufferBlocks[i].textureUnit;
  }
  return -1;
}


