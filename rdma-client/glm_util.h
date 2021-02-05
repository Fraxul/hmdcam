#pragma once
#include <boost/math/special_functions/fpclassify.hpp>
#include <glm/gtc/epsilon.hpp>

#define kXAxis 0
#define kYAxis 1
#define kZAxis 2

inline glm::vec3 transform(const glm::vec3& p, const glm::mat4& m) {
  glm::vec4 p4 = m * glm::vec4(p.x, p.y, p.z, 1.0f);
  return glm::vec3(p4) / p4.w;
}

template <typename T> inline bool is_valid_vector(const T& v) {
  for (size_t i = 0; i < v.length(); ++i)
    if (!boost::math::isfinite(v[i]))
      return false;

  return true;
}

template <typename T> inline bool vector_eq(const T& v1, const T& v2) {
  return glm::all(glm::epsilonEqual(v1, v2, 0.0001f));
}

static glm::vec3 linearToSRGB(glm::vec3 in) {
  glm::vec3 res;
  for (size_t i = 0; i < 3; ++i) {
    res[i] = in[i] <= 0.04045f ? (in[i] / 12.92f) : powf(((in[i] + 0.055f) / 1.055f), 2.4f);
  }
  return res;
}
static glm::vec4 linearToSRGB(glm::vec4 in) {
  return glm::vec4(linearToSRGB(glm::vec3(in)), in[3]);
}

static glm::vec3 srgbToLinear(glm::vec3 in) {
  glm::vec3 res;
  // equation straight out of 4.1.8 in the GL 4.1 spec
  for (size_t i = 0; i < 3; ++i) {
    if (in[i] < 0.0f) {
      res[i] = 0.0f;
    } else if (in[i] < 0.0031308f) {
      res[i] = 12.92f * in[i];
    } else if (in[i] < 1.0f) {
      res[i] = (1.055f * (powf(in[i], 0.41666f))) - 0.055f;
    } else {
      res[i] = 1.0f;
    }
  }
  return res;
}

static glm::vec4 srgbToLinear(glm::vec4 in) {
  return glm::vec4(srgbToLinear(glm::vec3(in)), in[3]);
}

