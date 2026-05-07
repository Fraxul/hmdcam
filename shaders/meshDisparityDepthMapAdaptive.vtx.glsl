#version 310 es

#include "MeshDisparityDepthMapUniformBlock.h"
#include "MeshDisparityDepthMapCommon.h"

// Adaptive-mesh path: each vertex carries its own grid coords and the disparity sampled
// at that corner. Variable-size patches share corner samples with same-level neighbors,
// so flat regions render as a continuous mesh; T-junctions at level transitions are
// expected and intentionally not stitched.
layout(location = 0) in uvec2 gridCoord;       // integer grid coords (W x H disparity grid)
layout(location = 1) in float disparityRawIn;  // raw disparity sampled at this corner

out V2F {
  vec2 texCoord;
} v2f;

void main() {
  int viewport = gl_InstanceID;
#ifndef SKIP_VIEWPORT_WRITE
  gl_ViewportIndex = viewport;
#endif

  float disparityRaw = (debugFixedDisparity >= 0) ? float(debugFixedDisparity) : disparityRawIn;
  float disparity = max(disparityRaw * disparityPrescale, (1.0f / 32.0f)); // prevent divide-by-zero

  vec2 textureCoordinates = vec2(gridCoord) * texCoordStep;
  vec2 gridCoordinates = vec2(gridCoord);
  gl_Position = modelViewProjection[viewport] * TransformToLocalSpace(vec4(textureCoordinates.xy, gridCoordinates.xy), disparity);

  v2f.texCoord = textureCoordinates;
}
