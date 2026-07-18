#version 310 es

#include "MeshDisparityDepthMapUniformBlock.h"

layout(location = 0) in V2F {
  vec2 texCoord;
} v2f;
uniform sampler2D imageTex;
uniform sampler2D distortionMap;

#if LEVEL_DEBUG
// Locations must match meshDisparityDepthMapAdaptive.vtx.glsl exactly: the
// linker pairs inter-stage varyings by location, not by name.
layout(location = 2) flat in uint v_debugLevelFlags;
layout(location = 3) in vec2 v_cellUV;
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

// Draw a constant-width bright line along any cell edge that was snapped to dRep -- i.e. an
// intentional depth-discontinuity split -- so it can be told apart from an unintended crack
// (which shows as a background-colored gap with no line). Returns coverage in [0,1].
float debugCrackLineCoverage(uint flags, vec2 cellUV) {
  const float kHalfWidthPx = 1.25;
  vec2 perPx = max(fwidth(cellUV), vec2(1e-6)); // cell-UV units per screen pixel
  float dLeft = cellUV.x / perPx.x;
  float dRight = (1.0 - cellUV.x) / perPx.x;
  float dTop = cellUV.y / perPx.y;
  float dBottom = (1.0 - cellUV.y) / perPx.y;
  float d = 1e9;
  if ((flags & 0x80u) != 0u) d = min(d, dLeft);   // kAdaptiveDebugLeftSnap
  if ((flags & 0x10u) != 0u) d = min(d, dRight);  // kAdaptiveDebugRightSnap
  if ((flags & 0x40u) != 0u) d = min(d, dTop);    // kAdaptiveDebugTopSnap
  if ((flags & 0x20u) != 0u) d = min(d, dBottom); // kAdaptiveDebugBottomSnap
  return 1.0 - smoothstep(kHalfWidthPx, kHalfWidthPx + 1.0, d);
}
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
  if (debugLevelColorMode > 0.5) {
    vec3 levelColor = debugLevelColor(v_debugLevelFlags);
    // Overlay a bright cyan line on intentionally-snapped (split) edges.
    float crack = debugCrackLineCoverage(v_debugLevelFlags, v_cellUV);
    outColor = vec4(mix(levelColor, vec3(0.0, 1.0, 1.0), crack), 1.0);
  }
#endif
}
