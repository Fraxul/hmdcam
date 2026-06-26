#version 310 es

#if DEPTH_BIAS
#extension GL_EXT_conservative_depth : require // enables layout(depth_greater) on gl_FragDepth
#endif

#include "MeshDisparityDepthMapUniformBlock.h"

in V2F {
  vec2 texCoord;
#if DEPTH_BIAS
  vec4 biasedClipPos;
#endif
} v2f;
uniform sampler2D imageTex;
uniform sampler2D distortionMap;

#if LEVEL_DEBUG
flat in uint v_debugLevelFlags;
#endif

layout(location = 0) out vec4 outColor;

#if LEVEL_DEBUG
// Per-cell diagnostic color: hue = chosen pyramid level, dimmed when this cell snapped an
// edge back to its representative disparity. Cracks show through as background-colored gaps.
vec3 debugLevelColor(uint flags) {
  const vec3 palette[5] = vec3[5](
    vec3(0.90, 0.12, 0.12),  // L0 red
    vec3(0.95, 0.60, 0.12),  // L1 orange
    vec3(0.20, 0.80, 0.25),  // L2 green
    vec3(0.20, 0.55, 0.95),  // L3 blue
    vec3(0.85, 0.30, 0.92)); // L4 magenta
  uint level = flags & 0xFu;
  vec3 c = palette[min(level, 4u)];
  bool snapped = (flags & 0xF0u) != 0u; // any of the four edges snapped to dRep
  return snapped ? c * 0.4 : c;
}
#endif

#if DEPTH_BIAS
// Conservative depth: the bias only ever pulls a fragment toward the camera, which under reverse-Z
// can only *increase* the written depth above gl_FragCoord.z. Declaring depth_greater lets the GPU
// keep hierarchical-Z rejection despite the gl_FragDepth write.
layout(depth_greater) out highp float gl_FragDepth;
#endif

void main() {
  // IMU depth timewarp: reproject the rectified sample coordinate by the inter-frame
  // homography before the distortion lookup, so the stale geometry samples the fresh color
  // where the same world point now appears. colorReprojection is identity when disabled.
  vec3 reproj = mat3(colorReprojection) * vec3(v2f.texCoord, 1.0);
  vec2 reprojectedTexCoord = reproj.xy / reproj.z;

  // Remap through OpenCV-generated distortion map
  vec2 distortionCoord = texture(distortionMap, reprojectedTexCoord).rg; // RG32F texture
  outColor = SAMPLE_CAMERA(distortionCoord);

#if LEVEL_DEBUG
  if (debugLevelColorMode > 0.5)
    outColor = vec4(debugLevelColor(v_debugLevelFlags), 1.0);
#endif

#if DEPTH_BIAS
  // Vulkan reverse-Z: window depth = clip.z / clip.w (NDC z already in [0,1]). Dividing the
  // perspective-correctly-interpolated clip z and w reproduces the exact screen-linear depth of
  // the biased (still planar) surface, which is what the rasterizer would emit for it.
  //
  // Guard the divide: once the eye-space bias pushes this surface to within viewZFightBiasMeters
  // of the camera, biasedClipPos.w crosses zero and z/w stops being a depth in [0,1] -- it flips
  // sign and, naively clamped, lands on the FAR plane (0.0) so the fragment fails the depth test
  // and vanishes. Pin to the near plane (1.0) whenever the biased point reaches or crosses it.
  // Both branches yield >= gl_FragCoord.z, so the depth_greater promise above still holds.
  gl_FragDepth = (v2f.biasedClipPos.w > 0.0)
      ? min(v2f.biasedClipPos.z / v2f.biasedClipPos.w, 1.0)
      : 1.0;
#endif
}
