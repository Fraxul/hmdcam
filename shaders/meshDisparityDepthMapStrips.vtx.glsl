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
#if DEPTH_BIAS
  vec2 biasedClipPosZW; // clip-space position translated toward the camera in eye space
#endif
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
  // Translate toward the camera by viewZFightBiasMeters of eye-space Z and re-project:
  // clip' = clip + dZ * Proj[:,2]. Biases only the depth buffer (see fragment shader).
  v2f.biasedClipPosZW = (gl_Position + viewZFightBiasMeters * projectionColumn2[viewport]).zw;
#endif
}
