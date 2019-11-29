#include "rhi/RHI.h"
#include "rhi/RHIShader.h"
#include <map>
#include <fstream>
#include <sstream>

#include <boost/functional/hash.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/lexical_cast.hpp>

static /*Cvar*/ bool rhi_dumpPreprocessedShaderSource = false;

const std::string& lookupShaderSourceCache(const std::string& path, bool forceReload) {
  static std::map<std::string, std::string> s_shaderSourceCache;

  std::string& source = s_shaderSourceCache[path];
  if (source.empty() || forceReload) {
    std::string fullPath = path; //FxVFS::expandPath(path); // Removed VFS dependency
    boost::interprocess::file_mapping fm(fullPath.c_str(), boost::interprocess::read_only);
    boost::interprocess::mapped_region mr(fm, boost::interprocess::read_only);

    source = std::string(static_cast<const char*>(mr.get_address()), mr.get_size());
  }
  return source;
}

RHIShader::RHIShader() : m_localWorkgroupSize(glm::uvec3(0)) {

}

RHIShader::~RHIShader() {

}

RHIShaderDescriptor::RHIShaderDescriptor() {
  rhi()->populateGlobalShaderDescriptorEnvironment(this);
}

RHIShaderDescriptor::RHIShaderDescriptor(const char* vertexShaderFilename, const char* fragmentShaderFilename, const RHIVertexLayout& vertexLayout) {
  rhi()->populateGlobalShaderDescriptorEnvironment(this);
  addSourceFile(RHIShaderDescriptor::kVertexShader, vertexShaderFilename);
  addSourceFile(RHIShaderDescriptor::kFragmentShader, fragmentShaderFilename);
  setVertexLayout(vertexLayout);
}

const char* RHIShaderDescriptor::nameForShadingUnit(ShadingUnit unit) {
  switch (unit) {
    case kVertexShader: return "vertex";
    case kGeometryShader: return "geometry";
    case kFragmentShader: return "fragment";
    case kTessEvaluationShader: return "tessellation eval";
    case kTessControlShader: return "tessellation control";
    case kComputeShader: return "compute";
    case kCommonHeader: return "common header";
    default: return NULL;
  }
}

void RHIShaderDescriptor::addSource(ShadingUnit unit, const char* text) {
  m_sources.push_back(Source(unit, std::string(), text));
}

void RHIShaderDescriptor::addSource(ShadingUnit unit, const std::string& text) {
  m_sources.push_back(Source(unit, std::string(), text));
}

void RHIShaderDescriptor::addSourceFile(ShadingUnit unit, const std::string& path) {
  m_sources.push_back(Source(unit, path, lookupShaderSourceCache(path)));
}

void RHIShaderDescriptor::reloadSources() const {
  for (size_t i = 0; i < m_sources.size(); ++i) {
    if (m_sources[i].filename.empty())
      continue;
    const_cast<RHIShaderDescriptor*>(this)->m_sources[i].text = lookupShaderSourceCache(m_sources[i].filename, true);
  }
}

void RHIShaderDescriptor::setFlag(const std::string& flagName, const std::string& flagValue) {
  m_flags[flagName] = flagValue;
}

void RHIShaderDescriptor::setFlag(const std::string& flagName, FxAtomicString flagValue) {
  m_flags[flagName] = flagValue.c_str();
}

void RHIShaderDescriptor::setFlag(const std::string& flagName, float flagValue) {
  m_flags[flagName] = boost::lexical_cast<std::string>(flagValue);
}

void RHIShaderDescriptor::setFlag(const std::string& flagName, int flagValue) {
  m_flags[flagName] = boost::lexical_cast<std::string>(flagValue);
}

void RHIShaderDescriptor::setFlag(const std::string& flagName, size_t flagValue) {
  m_flags[flagName] = boost::lexical_cast<std::string>(flagValue);
}

void RHIShaderDescriptor::setFlag(const std::string& flagName, bool flagValue) {
  m_flags[flagName] = flagValue ? "1" : "0";
}

void RHIShaderDescriptor::setVertexLayout(const RHIVertexLayout& layout) {
  m_vertexLayout = layout;
}

size_t RHIVertexLayoutElement::hash() const {
  size_t h = boost::hash_value(elementName);
  boost::hash_combine(h, offset);
  boost::hash_combine(h, stride);
  boost::hash_combine(h, arrayElementCount);
  boost::hash_combine(h, streamBufferIndex);
  boost::hash_combine(h, elementType);
  boost::hash_combine(h, elementFrequency);
  return h;
}

size_t RHIVertexLayout::hash() const {
  size_t h = boost::hash_value(elements.size());
  for (size_t i = 0; i < elements.size(); ++i) {
    boost::hash_combine(h, elements[i].hash());
  }
  return h;
}

size_t RHIShaderDescriptor::hash() const {
  size_t h = boost::hash_value("shader");
  for (size_t i = 0; i < m_sources.size(); ++i) {
    boost::hash_combine(h, m_sources[i].unit);
    boost::hash_combine(h, m_sources[i].text);
  }
  for (std::map<std::string, std::string>::const_iterator it = m_flags.begin(); it != m_flags.end(); ++it) {
    boost::hash_combine(h, it->first);
    boost::hash_combine(h, ':');
    boost::hash_combine(h, it->second);
    boost::hash_combine(h, '\n');
  }
  for (size_t i = 0; i < m_vertexLayout.elements.size(); ++i) {
    boost::hash_combine(h, m_vertexLayout.elements[i].hash());
  }
  return h;
}

std::map<RHIShaderDescriptor::ShadingUnit, std::string> RHIShaderDescriptor::preprocessSource() const {

  std::map<ShadingUnit, std::string> unitSources;
  // copy sources vector since we modify it
  std::vector<Source> sources_work = m_sources;

  for (size_t idx = 0; idx < sources_work.size(); ++idx) {
    const Source& source = sources_work[idx];
    if (source.unit >= kLastOpenGLShadingUnit)
      continue; // not relevant

    std::string& unit = unitSources[source.unit];

    size_t versionLoc = source.text.find("#version");
    if (versionLoc == std::string::npos) {
      fprintf(stderr, "Could not find version directive in shader:\n");
      dumpFormattedShaderSource(source.text.c_str());
      assert(false && "Internal shader compiler error: Could not find version directive.");
    }

    size_t versionLineEnd = source.text.find("\n", versionLoc);
    std::string versionLine = source.text.substr(versionLoc, (versionLineEnd + 1) - versionLoc);
    std::string restOfSource = source.text.substr(versionLineEnd + 1);

    if (unit.empty()) {
      // only put the version line, preprocessor defines, and comon declarations in once
      std::ostringstream outStream;
      outStream << versionLine << "\n";

      for (std::map<std::string, std::string>::const_iterator it = m_flags.begin(); it != m_flags.end(); ++it) {
        outStream << "#define " << it->first << " " << it->second << "\n";
      }
      outStream << "#define RHI_VERTEX_SHADER " << (source.unit == kVertexShader ? "1" : "0") << "\n";
      outStream << "#define RHI_FRAGMENT_SHADER " << (source.unit == kFragmentShader ? "1" : "0") << "\n";
      outStream << "#define RHI_COMPUTE_SHADER " << (source.unit == kComputeShader ? "1" : "0") << "\n";

      outStream << lookupShaderSourceCache("shaders/commonDeclarations.glsl") << "\n";

      // Put shader-specific common header sources in after the global commons
      for (size_t k = 0; k < sources_work.size(); ++k) {
        if (sources_work[k].unit != kCommonHeader)
          continue;
        const Source& commonSource = sources_work[k];
        outStream << commonSource.text << "\n";
      }

      // Append the common definitions to the shader sources for this unit as well. (it'll get picked up later in the loop)
      sources_work.push_back(Source(source.unit, "shaders/commonDefinitions.glsl", lookupShaderSourceCache("shaders/commonDefinitions.glsl")));
      unit += outStream.str();
    }

    unit += restOfSource;
  }

  std::map<ShadingUnit, std::string> res;
  for (std::map<ShadingUnit, std::string>::iterator unit_it = unitSources.begin(); unit_it != unitSources.end(); ++unit_it) {
    if (~unit_it->second.empty()) {
      res[unit_it->first] = unit_it->second;

      if (rhi_dumpPreprocessedShaderSource) {
        char namebuf[64];
        const char* unitname = NULL;
        switch (unit_it->first) {
          case kVertexShader: unitname = "vert"; break;
          case kGeometryShader: unitname = "geom"; break;
          case kFragmentShader: unitname = "frag"; break;
          case kTessEvaluationShader: unitname = "eval"; break;
          case kTessControlShader: unitname = "control"; break;
          case kComputeShader: unitname = "compute"; break;
          default: unitname = "unkn"; break;
        }

        sprintf(namebuf, "%0.16lx.%s.glsl", hash(), unitname);
        std::string outFilename = namebuf; //FxVFS::expandCachePath(std::string(namebuf));

        std::ofstream of(outFilename);
        of << res[unit_it->first];;
        printf("RHI: wrote preprocessed shader dump to %s\n", outFilename.c_str());
      }

    }
  }
  return res;
}

/*static*/ void RHIShaderDescriptor::dumpFormattedShaderSource(const char* source) {
  const char* p = source;
  int line = 1;
  while (*p) {
    fprintf(stderr, "%.4d | ", line++);
    while (*p) {
      fputc(*p, stderr);
      char c = *(p++);
      if (c == '\n') {
        break;
      }
    }
  }
  fputc('\n', stderr);
}

std::string RHIShaderDescriptor::getMetalSource() const {
  // No preprocessor define injection needed, the Metal framework handles that for us
  std::string res;
  for (size_t i = 0; i < m_sources.size(); ++i) {
    if (m_sources[i].unit == kMetalShader) {
      if (!res.empty())
        res += std::string("\n\n");

      res += m_sources[i].text;
    }
  }
  return res;
}

