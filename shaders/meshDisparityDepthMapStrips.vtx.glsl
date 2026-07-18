#version 310 es

#include "MeshDisparityDepthMapUniformBlock.h"
#include "MeshDisparityDepthMapCommon.h"

// Adaptive-strip path: local-space positions are prebaked by the CUDA strip-mesh builder
// (common/adaptiveStripMesh.cu), which merges runs of near-collinear point-path quads
// along each disparity row. This shader only projects the baked position and performs the
// same per-vertex color reprojection + distortion lookup as the point path.
layout(location = 0) in vec3 localPosition;
layout(location = 1) in vec2 rectifiedTexCoord; // unorm16 rectified UV, pre-distortion

uniform sampler2D distortionMap;

layout(location = 0) out V2F {
  vec2 texCoord;
} v2f;

void main()
{
  int viewport = gl_InstanceID;
#ifndef SKIP_VIEWPORT_WRITE
  gl_ViewportIndex = viewport;
#endif

  vec4 localPos = vec4(localPosition, 1.0f);
  if (debugFixedDisparity >= 0) {
    // Fixed-disparity debug / camera-failure fallback: the baked positions were built from
    // whatever is in the disparity buffer, so recompute from the rectified UV instead.
    float disparity = max(float(debugFixedDisparity) * disparityPrescale, (1.0f / 32.0f));
    localPos = TransformToLocalSpace(vec4(rectifiedTexCoord, 0.0f, 0.0f), disparity);
  }
  gl_Position = modelViewProjection[viewport] * localPos;

  // IMU depth timewarp + distortion correction: reproject the rectified texcoord by the
  // inter-frame homography, then remap through the distortion map. Same path as the point
  // renderer; vertex-rate distortion sampling error is <0.1px with our lens system.
  vec3 reproj = mat3(colorReprojection) * vec3(rectifiedTexCoord, 1.0);
  v2f.texCoord = textureLod(distortionMap, reproj.xy / reproj.z, 0.0).rg;

#if DEPTH_BIAS
  // Z-fight bias: push this vertex away from the camera by viewZFightBiasMeters of eye-space Z,
  // then fold the resulting depth change into gl_Position.z ALONE -- rescaled so gl_Position.z /
  // gl_Position.w equals the biased surface's NDC depth while w (hence screen x/y) is untouched.
  // The surface doesn't move, and depth still comes from the rasterizer (no gl_FragDepth write),
  // so early-Z/Hi-Z stays enabled. Reproduces the exact metric bias the old fragment path did.
  vec4 biasedClip = gl_Position - viewZFightBiasMeters * projectionColumn2[viewport];
  gl_Position.z = biasedClip.z * gl_Position.w / biasedClip.w;
#endif
}
