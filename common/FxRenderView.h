#pragma once
//#include "FxFrustum.h"
#include "rhi/RHI.h"
#include <glm/glm.hpp>

struct FxRenderView {
  glm::vec3 worldEyePosition;
  glm::vec3 worldViewVector;
  glm::mat4 viewMatrix;
  glm::mat4 projectionMatrix;
  glm::mat4 viewProjectionMatrix; // composite
  float fov; // X FOV, in degrees
  float fovY; // Y FOV, in degrees
  float aspect; // aspect ratio
  float zNear, zFar; // world space projection depth parameters
  float zMin, zMax; // hardware depth range, usually 0.0f - 1.0f

  //FxFrustum frustum;
  RHIRect viewportRect;

  // call after adjusting projectionMatrix or viewMatrix:
  // recomputes viewProjectionMatrix and frustum
  void recomputeFrustumData();
};

