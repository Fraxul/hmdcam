#pragma once
#include "FxAtomicString.h"
#include "rhi/RHIComputePipeline.h"
#include "rhi/RHIObject.h"
#include "rhi/RHIRenderPipeline.h"
#include <stddef.h>
#include <stdint.h>
#include <glm/glm.hpp>
#include <boost/container/static_vector.hpp>
#include <map>
#include <string>
#include <vector>

const std::string& lookupShaderSourceCache(const std::string& path, bool forceReload = false);

enum RHIVertexElementType : unsigned char {
  kVertexElementTypeNone,
  kVertexElementTypeFloat1,
  kVertexElementTypeFloat2,
  kVertexElementTypeFloat3,
  kVertexElementTypeFloat4,

  kVertexElementTypeHalf1,
  kVertexElementTypeHalf2,
  // 6 byte fetch is a performance hit on some hardware, we don't support it.
  kVertexElementTypeHalf4,

  kVertexElementTypeInt1,
  kVertexElementTypeInt2,
  kVertexElementTypeInt3,
  kVertexElementTypeInt4,

  kVertexElementTypeUInt1,
  kVertexElementTypeUInt2,
  kVertexElementTypeUInt3,
  kVertexElementTypeUInt4,

  kVertexElementTypeUByte1,
  kVertexElementTypeUByte2,
  kVertexElementTypeUByte3,
  kVertexElementTypeUByte4,

  kVertexElementTypeByte1N,
  kVertexElementTypeByte2N,
  kVertexElementTypeByte3N,
  kVertexElementTypeByte4N,

  kVertexElementTypeUByte1N,
  kVertexElementTypeUByte2N,
  kVertexElementTypeUByte3N,
  kVertexElementTypeUByte4N,

  kVertexElementTypeUShort1,
  kVertexElementTypeUShort2,
  kVertexElementTypeUShort4,

  kVertexElementTypeUShort1N,
  kVertexElementTypeUShort2N,
  kVertexElementTypeUShort4N,

  kVertexElementTypeShort1N,
  kVertexElementTypeShort2N,
  kVertexElementTypeShort4N,
};

enum RHIVertexElementFrequency : unsigned char {
  kVertexElementFrequencyVertex,
  kVertexElementFrequencyInstance,
  kVertexElementFrequencyConstant,
};

class RHI;

class RHIShader : public RHIObject {
public:
  typedef boost::intrusive_ptr<RHIShader> ptr;
  RHIShader();
  virtual ~RHIShader();

  // only applies to compute shaders
  const glm::uvec3& localWorkgroupSize() const { return m_localWorkgroupSize; }

protected:
  friend class RHI;
  glm::uvec3 m_localWorkgroupSize;
  std::map<size_t, RHIRenderPipeline::ptr> m_renderPipelineCache;
  RHIComputePipeline::ptr m_computePipelineCache;
};

struct RHIVertexLayoutElement {
  RHIVertexLayoutElement() : offset(0), stride(0), arrayElementCount(1), streamBufferIndex(0), elementType(kVertexElementTypeNone), elementFrequency(kVertexElementFrequencyVertex) {}
  RHIVertexLayoutElement(uint8_t streamBufferIndex_, RHIVertexElementType elementType_, const FxAtomicString name_, uint16_t offset_, uint16_t stride_, uint8_t arrayElementCount_ = 1,
    RHIVertexElementFrequency elementFrequency_ = kVertexElementFrequencyVertex) : elementName(name_), offset(offset_), stride(stride_), arrayElementCount(arrayElementCount_), streamBufferIndex(streamBufferIndex_), elementType(elementType_), elementFrequency(elementFrequency_) {}

  size_t hash() const;

  FxAtomicString elementName;
  uint16_t offset;
  uint16_t stride;
  uint8_t arrayElementCount;
  uint8_t streamBufferIndex;
  RHIVertexElementType elementType;
  RHIVertexElementFrequency elementFrequency;
};

struct RHIVertexLayout {
  RHIVertexLayout() {}
  RHIVertexLayout(const std::initializer_list<RHIVertexLayoutElement>& el_) : elements(el_.begin(), el_.end()) {}

  boost::container::static_vector<RHIVertexLayoutElement, 16> elements;

  size_t hash() const;
};

class RHIShaderDescriptor {
public:
  RHIShaderDescriptor();

  // Common-case constructor
  RHIShaderDescriptor(const char* vertexShaderFilename, const char* fragmentShaderFilename, const RHIVertexLayout& vertexLayout);

  enum ShadingUnit {
    kVertexShader,
    kGeometryShader,
    kFragmentShader,
    kTessEvaluationShader,
    kTessControlShader,

    kComputeShader,

    kLastOpenGLShadingUnit,

    // not a real unit; adding kCommonHeader source puts it in front of the actual source for every unit.
    kCommonHeader,

    kMetalShader
  };

  static const char* nameForShadingUnit(ShadingUnit);

  void addSource(ShadingUnit unit, const char*);
  void addSource(ShadingUnit unit, const std::string&);
  void addSourceFile(ShadingUnit unit, const std::string&);

  void reloadSources() const;

  void setFlag(const std::string&, const char*);
  void setFlag(const std::string&, const std::string&);
  void setFlag(const std::string&, FxAtomicString);
  void setFlag(const std::string&, bool);
  void setFlag(const std::string&, size_t);
  void setFlag(const std::string&, int);
  void setFlag(const std::string&, float);

  void setVertexLayout(const RHIVertexLayout&);

  struct Source {
    Source() {}

    Source(ShadingUnit _unit, const std::string& _filename, const std::string& _text) : unit(_unit), filename(_filename), text(_text) {}
    ShadingUnit unit;
    std::string filename;
    std::string text;
  };

  size_t hash() const;

  // shared utility
  const RHIVertexLayout& vertexLayout() const { return m_vertexLayout; }
  std::map<ShadingUnit, std::string> preprocessSource() const;
  static void dumpFormattedShaderSource(const char*);
  const std::map<std::string, std::string>& flags() const { return m_flags; }
  std::string getMetalSource() const;

  void debugDumpSourceMap() const;

protected:
  std::vector<Source> m_sources;
  std::map<std::string, std::string> m_flags;
  RHIVertexLayout m_vertexLayout;
};

