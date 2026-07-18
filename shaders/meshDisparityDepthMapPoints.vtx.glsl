#version 310 es

#include "MeshDisparityDepthMapUniformBlock.h"
#include "MeshDisparityDepthMapCommon.h"

layout(location = 0) in uvec2 disparitySampleCoordinates; // integer texels, fixed to the left-top value
layout(location = 1) in uvec2 quadCoordOffset; // 0...1, varies over the quad

uniform highp usampler2D disparityTex;
uniform sampler2D distortionMap;
float sampleDisparity(ivec2 mipCoords) {
  return float(texelFetch(disparityTex, mipCoords, 0).r);
}

layout(location = 0) out V2F {
  vec2 texCoord;
} v2f;

void main()
{
  int viewport = gl_InstanceID;
#ifndef SKIP_VIEWPORT_WRITE
  gl_ViewportIndex = viewport;
#endif

  if (any(notEqual(clamp(vec2(disparitySampleCoordinates.xy), trim_minXY, trim_maxXY), vec2(disparitySampleCoordinates.xy)))) {
    // Trimmed -- collapse primitive in clip space
    gl_Position = vec4(0.0f);
    return;
  }

  float disparityRaw = 0.0f;
  if (debugFixedDisparity >= 0) {
    disparityRaw = float(debugFixedDisparity);
  } else {
    disparityRaw = sampleDisparity(ivec2(disparitySampleCoordinates));
  }

  vec2 textureCoordinates = vec2(disparitySampleCoordinates) * texCoordStep + (vec2(quadCoordOffset) * texCoordStep * pointScale);

  // IMU depth timewarp + distortion correction: reproject the rectified texcoord by the inter-frame homography, then remap through the distortion map.
  // The distortion map is smooth enough with our lens system that the error from sampling it at vertex rate vs. fragment rate is <0.1px.
  vec3 reproj = mat3(colorReprojection) * vec3(textureCoordinates, 1.0);
  vec2 cameraSampleCoordinates = textureLod(distortionMap, reproj.xy / reproj.z, 0.0).rg;

  if (disparityRaw > float(maxValidDisparityRaw)) {
    // Invalid disparity, so we discard this point.
    gl_Position = vec4(0.0f);
  } else {
    float disparity = max(disparityRaw * disparityPrescale, (1.0f / 32.0f)); // prescale and prevent divide-by-zero
    vec2 gridCoordinates = vec2(disparitySampleCoordinates) + (vec2(quadCoordOffset) * pointScale);
    gl_Position = modelViewProjection[viewport] * TransformToLocalSpace(vec4(textureCoordinates.xy, gridCoordinates.xy), disparity);
  }

#if DEPTH_BIAS
  // Z-fight bias: push this vertex away from the camera by viewZFightBiasMeters of eye-space Z,
  // then fold the resulting depth change into gl_Position.z ALONE -- rescaled so gl_Position.z /
  // gl_Position.w equals the biased surface's NDC depth while w (hence screen x/y) is untouched.
  // No gl_FragDepth write, so early-Z/Hi-Z stays enabled. The collapsed/invalid branches above
  // set gl_Position = 0; biasedClip.w = viewZFightBiasMeters there, so the degenerate vertex stays
  // finite (and is culled anyway).
  vec4 biasedClip = gl_Position - viewZFightBiasMeters * projectionColumn2[viewport];
  gl_Position.z = biasedClip.z * gl_Position.w / biasedClip.w;
#endif

  v2f.texCoord = cameraSampleCoordinates;
}

