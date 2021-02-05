#pragma once
#include "FxRenderView.h"
#include <glm/glm.hpp>
#include <vector>

class FxCamera {
public:
  FxCamera();
  ~FxCamera();

  // Rotate the targetPosition around the position
  void spin(const glm::vec2& delta);
  // Rotate the position around the targetPosition; rotation axes aligned with the view plane
  void tumble(const glm::vec2& delta);
  // Move the position and targetPosition in the view plane
  void track(const glm::vec2& delta);
  // Zoom the camera in and out (move position along the vector between it and targetPosition)
  void dolly(float delta);

  std::vector<FxRenderView> toRenderViews(float renderTargetAspectRatio, bool forStereoView) const;
  FxRenderView toRenderView(float renderTargetAspectRatio, int forStereoEye = 0) const;

  glm::vec3 position() const;
  void setPosition(const glm::vec3& position) { m_position = position; }

  const glm::vec3& targetPosition() const { return m_targetPosition; }
  void setTargetPosition(const glm::vec3& targetPosition) { m_targetPosition = targetPosition; }

  glm::vec3 lookVector() const;

  const glm::vec3& upVector() const { return m_upVec; }
  void setUpVector(const glm::vec3& upVector) { m_upVec = upVector; }

  float fieldOfView() const { return m_fovX; }
  void setFieldOfView(float fovX) { m_fovX = fovX; }

  float zNear() const { return m_zNear; }
  void setZNear(float zNear) { m_zNear = zNear; }

  float zFar() const { return m_zFar; }
  void setZFar(float zFar) { m_zFar = zFar; }

  void setUseHMDControl(bool value) { m_useHMDControl = value; }

protected:
  glm::vec3 m_position;
  glm::vec3 m_targetPosition;
  glm::vec3 m_upVec;

  float m_fovX;
  float m_zNear, m_zFar;

  bool m_useHMDControl;
};

