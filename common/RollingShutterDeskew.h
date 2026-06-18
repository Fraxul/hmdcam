#pragma once
#include <cmath>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

// Convention-critical primitives shared between the live rolling-shutter de-skew
// (CameraSystem::processFrame) and the offline visualization harness
// (calibration/rollingShutterDeskewTest). Keeping them in one header guarantees the
// two code paths apply the exp-map and the IMU-to-camera rotation identically, so the
// harness is a faithful test of what the live pipeline does.

namespace RollingShutterDeskew {

// Exponential map: rotation vector (axis * angle, radians) -> 3x3 rotation matrix,
// row-major out[r*3 + c]. Mirrors expSO3() in
// calibration/cameraImuCalibration/Conventions.h: form the unit quaternion (w, xyz)
// with the small-angle Taylor branch for the vector scale, then expand to a matrix.
// Using the same construction as the calibrator guarantees the de-skew integrates
// rotation with the identical sign/convention the calibration was fit under.
inline void rotationVectorToMatrix(const glm::vec3& rotationVector, float outRowMajor[9]) {
  const float thetaSq = glm::dot(rotationVector, rotationVector);
  const float theta = std::sqrt(thetaSq);
  const float halfAngle = 0.5f * theta;

  // vectorScale = sin(theta/2)/theta, with the theta->0 Taylor limit (1/2 - theta^2/48).
  float vectorScale;
  if (theta > 1.0e-8f)
    vectorScale = std::sin(halfAngle) / theta;
  else
    vectorScale = 0.5f - (thetaSq * (1.0f / 48.0f));

  const float w = std::cos(halfAngle);
  const float x = rotationVector.x * vectorScale;
  const float y = rotationVector.y * vectorScale;
  const float z = rotationVector.z * vectorScale;

  const float xx = x * x, yy = y * y, zz = z * z;
  const float xy = x * y, xz = x * z, yz = y * z;
  const float wx = w * x, wy = w * y, wz = w * z;

  outRowMajor[0] = 1.0f - (2.0f * (yy + zz));
  outRowMajor[1] = 2.0f * (xy - wz);
  outRowMajor[2] = 2.0f * (xz + wy);
  outRowMajor[3] = 2.0f * (xy + wz);
  outRowMajor[4] = 1.0f - (2.0f * (xx + zz));
  outRowMajor[5] = 2.0f * (yz - wx);
  outRowMajor[6] = 2.0f * (xz - wy);
  outRowMajor[7] = 2.0f * (yz + wx);
  outRowMajor[8] = 1.0f - (2.0f * (xx + yy));
}

// Exponential map: rotation vector (axis * angle, radians) -> unit quaternion. Same
// construction as rotationVectorToMatrix (and expSO3 in the calibrator's Conventions.h);
// provided for callers that compose/store rotations as quaternions (e.g. the per-camera
// inter-frame rotation used by IMU depth timewarp) rather than as the kernel's row-major
// matrix buffer.
inline glm::quat rotationVectorToQuaternion(const glm::vec3& rotationVector) {
  const float thetaSq = glm::dot(rotationVector, rotationVector);
  const float theta = std::sqrt(thetaSq);
  const float halfAngle = 0.5f * theta;

  float vectorScale;
  if (theta > 1.0e-8f)
    vectorScale = std::sin(halfAngle) / theta;
  else
    vectorScale = 0.5f - (thetaSq * (1.0f / 48.0f));

  return glm::quat(std::cos(halfAngle),
    rotationVector.x * vectorScale,
    rotationVector.y * vectorScale,
    rotationVector.z * vectorScale);
}

// Apply R_imu_to_cam (as loaded into a glm::mat3 by glmMat3FromCVMatrix) to a
// body-frame vector. glmMat3FromCVMatrix assigns cv(r,c) to glm index [r][c]; since
// glm is column-major, indexing the object back as [r][c] yields the true cv(r,c)
// element, but glm's operator* would silently apply the matrix transposed. We index
// [r][c] by hand here to apply the rotation in the intended (cv) orientation. The
// harness must therefore also load R_imu_to_cam via glmMat3FromCVMatrix.
inline glm::vec3 applyImuToCameraRotation(const glm::mat3& imuToCameraRotation, const glm::vec3& v) {
  return glm::vec3(
    (imuToCameraRotation[0][0] * v.x) + (imuToCameraRotation[0][1] * v.y) + (imuToCameraRotation[0][2] * v.z),
    (imuToCameraRotation[1][0] * v.x) + (imuToCameraRotation[1][1] * v.y) + (imuToCameraRotation[1][2] * v.z),
    (imuToCameraRotation[2][0] * v.x) + (imuToCameraRotation[2][1] * v.y) + (imuToCameraRotation[2][2] * v.z));
}

} // namespace RollingShutterDeskew
