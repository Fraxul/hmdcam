#version 310 es

#include "MeshDisparityDepthMapUniformBlock.h"
#include "MeshDisparityDepthMapCommon.h"

// Adaptive-mesh path: each vertex carries its own grid coords and the disparity sampled
// at that corner. Variable-size patches share corner samples with same-level neighbors,
// so flat regions render as a continuous mesh; T-junctions at level transitions are
// expected and intentionally not stitched.
layout(location = 0) in uvec2 gridCoord;       // q12.4 grid coords (pixel * 16, W x H disparity grid)
layout(location = 1) in float disparityRawIn;  // raw disparity sampled at this corner
#if LEVEL_DEBUG
layout(location = 2) in uint debugLevelFlags;  // bits 0-3 level, 4-7 edge snaps, 8-9 corner index
// Inter-stage varyings MUST carry explicit, matching locations: each stage is
// compiled in isolation and the linker pairs them by location, not name. The
// V2F block below takes location 0 (and 1 under DEPTH_BIAS), so the loose
// varyings start at 2. Keep these in sync with the fragment shader.
layout(location = 2) flat out uint v_debugLevelFlags;
layout(location = 3) out vec2 v_cellUV;  // (0,0)=TL .. (1,1)=BR, interpolated across the quad
#endif

// Matches kAdaptiveMeshGridFracBits / kAdaptiveMeshGridScale in depthMeshAdaptive.h.
const float kGridScaleInv = 1.0 / 16.0;

layout(location = 0) out V2F {
  vec2 texCoord;
#if DEPTH_BIAS
  vec4 biasedClipPos; // clip-space position translated toward the camera in eye space
#endif
} v2f;

void main() {
  int viewport = gl_InstanceID;
#ifndef SKIP_VIEWPORT_WRITE
  gl_ViewportIndex = viewport;
#endif

  float disparityRaw = (debugFixedDisparity >= 0) ? float(debugFixedDisparity) : disparityRawIn;
  float disparity = max(disparityRaw * disparityPrescale, (1.0f / 32.0f)); // prevent divide-by-zero

  vec2 gridCoordinates = vec2(gridCoord) * kGridScaleInv;
  vec2 textureCoordinates = gridCoordinates * texCoordStep;
  gl_Position = modelViewProjection[viewport] * TransformToLocalSpace(vec4(textureCoordinates.xy, gridCoordinates.xy), disparity);

#if DEPTH_BIAS
  // Translate this vertex toward the camera by viewZFightBiasMeters of eye-space Z and re-project:
  // clip' = Proj * (eye + dZ) = clip + dZ * Proj[:,2]. Carried to the fragment shader to bias only
  // the depth buffer; gl_Position itself is untouched so the rendered surface does not move.
  v2f.biasedClipPos = gl_Position + viewZFightBiasMeters * projectionColumn2[viewport];
#endif

  v2f.texCoord = textureCoordinates;

#if LEVEL_DEBUG
  v_debugLevelFlags = debugLevelFlags;
  uint cornerId = (debugLevelFlags >> 8u) & 3u;
  v_cellUV = vec2(float(cornerId & 1u), float((cornerId >> 1u) & 1u));
#endif
}
