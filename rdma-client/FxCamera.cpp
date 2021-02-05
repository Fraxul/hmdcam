#include "FxCamera.h"
#include "glm_util.h"
#include <glm/gtc/constants.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/transform.hpp>
#define _USE_MATH_DEFINES
#include <math.h>

#if 0
#ifdef _WIN32
#include "openvr.h"
#else
#include <OpenVR/openvr.h>
#endif
#endif

static float r_zNear = 2.0f;
static float r_stereoSeparation = 0.5f;
static float hmd_worldUnitsPerMeter = 100.0f;
static bool hmd_pitchLock = true;

FxCamera::FxCamera() :
  m_position(glm::vec3(0, 0, -15)),
  m_targetPosition(glm::vec3(0, 0, 0)),
  m_upVec(glm::vec3(0, 1, 0)),
  m_fovX(45.0f),
  m_zNear(1.0f), m_zFar(10000.0f),
  m_useHMDControl(false) {

}

FxCamera::~FxCamera() {

}

glm::vec3 FxCamera::position() const {
  return m_position;
}

glm::vec3 FxCamera::lookVector() const {
  return glm::normalize(targetPosition() - position());
}

void FxCamera::spin(const glm::vec2& delta) {
  glm::vec3 aim = glm::normalize(m_targetPosition - m_position);
  glm::vec3 vpRight = glm::normalize(glm::cross(aim, m_upVec));
  glm::vec3 vpUp = glm::normalize(glm::cross(vpRight, aim));

  glm::vec3 positionVector = m_targetPosition - m_position;

  if (delta[0] != 0.0)
    positionVector = glm::rotate(positionVector, glm::radians(-delta[0] * 0.1f), vpUp);
  // Lock rotation to the XZ plane if we're attached to an HMD.
  if (!(m_useHMDControl && hmd_pitchLock)) {
    if (delta[1] != 0.0)
      positionVector = glm::rotate(positionVector, glm::radians(-delta[1] * 0.1f), vpRight);
  }

  m_targetPosition = m_position + positionVector;
}

void FxCamera::tumble(const glm::vec2& delta) {
  glm::vec3 aim = glm::normalize(m_targetPosition - m_position);
  glm::vec3 vpRight = glm::normalize(glm::cross(aim, m_upVec));
  glm::vec3 vpUp = glm::normalize(glm::cross(vpRight, aim));

  glm::vec3 positionVector = m_position - m_targetPosition;

  if (delta[0] != 0.0)
    positionVector = glm::rotate(positionVector, glm::radians(delta[0] * -0.5f), vpUp);
  // Lock rotation to the XZ plane if we're attached to an HMD.
  if (!(m_useHMDControl && hmd_pitchLock)) {
    if (delta[1] != 0.0)
      positionVector = glm::rotate(positionVector, glm::radians(delta[1] * -0.5f), vpRight);
  }

  m_position = m_targetPosition + positionVector;

}

void FxCamera::track(const glm::vec2& delta) {
  if (delta[0] == 0.0 && delta[1] == 0.0)
    return;

  // The 'delta' vector is in fractions of the viewport X/Y that the mouse moved
  // so delta X = 1.0 means that the mouse moved all the way across the viewport left to right.
  // (this is designed so that the Camera doesn't ever need to know the pixel dimensions of the viewport
  // it's being rendered into.)

  // We'll be doing the rest of the math in NDC, which has a -1 to 1 range; scale out the delta to
  // fill that.
  glm::vec2 ndcDelta = delta * 2.0f;

  // Compute the view plane x and y vectors
  glm::vec3 aim = glm::normalize(m_targetPosition - m_position);
  glm::vec3 vpRight = glm::normalize(glm::cross(aim, m_upVec));
  glm::vec3 vpUp = glm::normalize(glm::cross(vpRight, aim));

  // Compute appropriate lengths for the right and up vectors.
  // This doesn't need the actual exact aspect ratio (which we don't have), so we just guess 16/9 and call it good.
  const FxRenderView& renderView = toRenderView(16.0f / 9.0f);

  // project the target position into NDC
  glm::vec3 ndcTarget = transform(m_targetPosition, renderView.viewProjectionMatrix);

  // project the endpoints of the unit right and up vectors (relative to the target position) as well
  glm::vec3 ndcRight = transform(m_targetPosition + vpRight, renderView.viewProjectionMatrix);
  glm::vec3 ndcUp = transform(m_targetPosition + vpUp, renderView.viewProjectionMatrix);

  // the lengths of the vectors from the target to the right and up coordinates in NDC space are used to scale
  // the corresponding vectors in world space
  vpRight /= glm::length(ndcRight - ndcTarget);
  vpUp /= glm::length(ndcUp - ndcTarget);

  // now move the world space positions (camera and target) by the product of the NDC delta vector and the scaled WS right/up vectors.
  glm::vec3 move = (ndcDelta[0] * vpRight) + (ndcDelta[1] * vpUp);

  m_position += move;
  m_targetPosition += move;
}

void FxCamera::dolly(float delta) {
  glm::vec3 aim = glm::normalize(m_targetPosition - m_position);

  m_position += (aim * delta);

  if (glm::length(m_position - m_targetPosition) < 1.0f) {
    // too close for comfort, reset to a unit away from targetPosition so our
    // coordinate system doesn't collapse into a singularity
    m_position = m_targetPosition - aim;
  }
}

std::vector<FxRenderView> FxCamera::toRenderViews(float renderTargetAspectRatio, bool forStereoView) const {
  std::vector<FxRenderView> res;
  if (forStereoView)
    renderTargetAspectRatio /= 2.0f; // SxS view has only half the effective aspect ratio per-eye

  res.push_back(toRenderView(renderTargetAspectRatio, 0));
  if (forStereoView)
    res.push_back(toRenderView(renderTargetAspectRatio, 1));

  return res;
}

#if 0
glm::mat4 convertMatrix(vr::HmdMatrix44_t mat) {
  return glm::mat4(
    mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
    mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
    mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
    mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]);
}

glm::mat4 convertMatrix(vr::HmdMatrix34_t mat) {
  return glm::mat4(
    mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0f,
    mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0f,
    mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0f,
    mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0f);
}
#endif

FxRenderView FxCamera::toRenderView(float renderTargetAspectRatio, int forStereoEye) const {
  FxRenderView res;
  res.worldEyePosition = this->position();
  res.worldViewVector = lookVector();

  glm::vec3 target = targetPosition();

  if (forStereoEye == 1 && !m_useHMDControl) {
    // Apply offset for the right eye
    glm::vec3 stereoOffset = glm::normalize(glm::cross(res.worldViewVector, m_upVec)) * r_stereoSeparation;
    res.worldEyePosition += stereoOffset;
    target += stereoOffset;
  }


  res.viewMatrix = glm::lookAtRH(res.worldEyePosition, /*center=*/target, m_upVec);

  res.fov = m_fovX;
  res.fovY = m_fovX / renderTargetAspectRatio;
  res.aspect = renderTargetAspectRatio;

  glm::vec2 projScale = glm::vec2(
    1.0f / glm::tan(glm::radians(res.fov  * 0.5f)),
    1.0f / glm::tan(glm::radians(res.fovY * 0.5f)) );
  glm::vec2 projOffset = glm::vec2(0.0f);

#if 1
  // Right-handed infinite-Z far plane
  // TODO option to use this
  res.projectionMatrix = glm::mat4(
    glm::vec4(projScale.x,    0.0f,        projOffset.x,     0.0f),
    glm::vec4(0.0f,           projScale.y, projOffset.y,     0.0f),
    glm::vec4(0.0f,           0.0f,        0.0f,            -1.0f),
    glm::vec4(0.0f,           0.0f,        r_zNear,          0.0f)
  );

#else
  // bounded-Z far plane
  float z1 = r_zNear.value() / (r_zNear.value() - m_zFar);
  float z2 = -m_zFar * z1;
  res.projectionMatrix = glm::mat4(
    glm::vec4(projScale.x,    0.0f,        projOffset.x,  0.0f),
    glm::vec4(0.0f,           projScale.y, projOffset.y,  0.0f),
    glm::vec4(0.0f,           0.0f,        z1,           -1.0f),
    glm::vec4(0.0f,           0.0f,        z2,            0.0f)
  );
#endif

  res.zNear = r_zNear;
  res.zFar = m_zFar;
  res.zMin = 0.0f;
  res.zMax = 1.0f;
#if 0
  if (m_useHMDControl) {
    vr::EVREye eEye = (vr::EVREye) forStereoEye ? vr::Eye_Right : vr::Eye_Left;

    {
      // Build a right-handed infinite-perspective projection matrix with the raw projection coefficients from OpenVR
      float projLeft, projRight, projTop, projBottom;
      FxEngine::vrSystem()->GetProjectionRaw(eEye, &projLeft, &projRight, &projTop, &projBottom);

      float idx = 1.0f / (projRight - projLeft);
      float idy = 1.0f / (projBottom - projTop);
      float sx = projRight + projLeft;
      float sy = projBottom + projTop;
 
       res.projectionMatrix = glm::mat4(
        2.0f*idx,  0.0f,      0.0f,             0.0f,
        0.0f,      2.0f*idy,  0.0f,             0.0f,
        sx*idx,    sy*idy,    0.0f,            -1.0f,
        0.0f,      0.0f,      r_zNear.value(),  0.0f);

    }

    glm::mat4 eyeToHead = convertMatrix(FxEngine::vrSystem()->GetEyeToHeadTransform(eEye));

    vr::TrackedDevicePose_t hmdPose;
    vr::VRCompositor()->GetLastPoseForTrackedDeviceIndex(vr::k_unTrackedDeviceIndex_Hmd, &hmdPose, NULL);

    glm::mat4 deviceToAbsolute = convertMatrix(hmdPose.mDeviceToAbsoluteTracking);

    glm::mat4 absoluteToDevice = glm::inverse(deviceToAbsolute);
    for (size_t i = 0; i < 3; ++i) {
      // scale view movement
      absoluteToDevice[3][i] *= hmd_worldUnitsPerMeter;
    }

    // Stereo offset is the inverse of the eyeToHead matrix, which is just an X-axis transform
    glm::vec3 stereoOffset = glm::vec3(-eyeToHead[3][0] * hmd_worldUnitsPerMeter, 0.0f, 0.0f);

    // HMD rotation
    glm::mat4 hmdViewOffset = glm::translate(stereoOffset) * absoluteToDevice;

    glm::vec3 flatViewVector = lookVector();
    if (hmd_pitchLock) {
      flatViewVector[1] = 0.0f;
      flatViewVector = glm::normalize(flatViewVector);
    }
    res.viewMatrix = hmdViewOffset * glm::lookAtRH(res.worldEyePosition, res.worldEyePosition + flatViewVector, m_upVec);

    // compute new worldEyePosition with hmd position offset
    glm::mat4 invView = glm::inverse(res.viewMatrix);
    res.worldEyePosition = glm::vec3(invView[3][0], invView[3][1], invView[3][2]);
  }
#endif

  res.recomputeFrustumData();
  return res;
}

