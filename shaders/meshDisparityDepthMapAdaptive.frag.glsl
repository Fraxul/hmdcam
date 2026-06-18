#version 310 es

#include "MeshDisparityDepthMapUniformBlock.h"

in V2F {
  vec2 texCoord;
} v2f;
uniform sampler2D imageTex;
uniform sampler2D distortionMap;

layout(location = 0) out vec4 outColor;

void main() {
  // IMU depth timewarp: reproject the rectified sample coordinate by the inter-frame
  // homography before the distortion lookup, so the stale geometry samples the fresh color
  // where the same world point now appears. colorReprojection is identity when disabled.
  vec3 reproj = mat3(colorReprojection) * vec3(v2f.texCoord, 1.0);
  vec2 reprojectedTexCoord = reproj.xy / reproj.z;

  // Remap through OpenCV-generated distortion map
  vec2 distortionCoord = texture(distortionMap, reprojectedTexCoord).rg; // RG32F texture
  outColor = SAMPLE_CAMERA(distortionCoord);
}
