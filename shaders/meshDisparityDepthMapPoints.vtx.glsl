#version 310 es

#include "MeshDisparityDepthMapUniformBlock.h"

layout(location = 0) in uvec2 disparitySampleCoordinates; // integer texels, fixed to the left-top value
layout(location = 1) in uvec2 quadCoordOffset; // 0...1, varies over the quad

#if DISPARITY_USE_FP16
uniform sampler2D disparityTex;
float sampleDisparity(ivec2 mipCoords, int level) {
  return texelFetch(disparityTex, mipCoords, level).r;
}
#else
uniform highp usampler2D disparityTex;
float sampleDisparity(ivec2 mipCoords, int level) {
  return float(texelFetch(disparityTex, mipCoords, level).r);
}
#endif

out V2F {
  vec2 texCoord;
} v2f;

vec4 TransformToLocalSpace( float x, float y, float fDisp ) {

  float lz = Q11 * CameraDistanceMeters / (fDisp * mogrify.x);
  float ly = ((y * mogrify.y) + Q7) / Q11;
  float lx = ((x * mogrify.x) + Q3) / Q11;
  lx *= lz;
  ly *= lz;
  return R1inv * vec4(lx, -ly, -lz, 1.0f);
}

void main()
{
  int viewport = gl_InstanceID;
  gl_ViewportIndex = viewport;

  if (any(notEqual(clamp(vec2(disparitySampleCoordinates.xy), trim_minXY, trim_maxXY), vec2(disparitySampleCoordinates.xy)))) {
    // Trimmed -- collapse primitive in clip space
    gl_Position = vec4(0.0f);
    return;
  }

  float disparityRaw = 0.0f;
  if (debugFixedDisparity >= 0) {
    disparityRaw = float(debugFixedDisparity);

  } else {
    ivec2 mipCoords = ivec2(disparitySampleCoordinates);
    // Walk the mip chain to find a valid disparity value at this location
    for (int level = 0; level < disparityTexLevels; ++level) {
      disparityRaw = sampleDisparity(mipCoords, level);
      if (disparityRaw <= float(maxValidDisparityRaw))
        break;
      mipCoords = mipCoords >> 1;
    }
  }

  float disparity = max(disparityRaw * disparityPrescale, (1.0f / 32.0f)); // prescale and prevent divide-by-zero
  vec2 gridCoordinates = vec2(disparitySampleCoordinates) + (vec2(quadCoordOffset) * pointScale);
  gl_Position = modelViewProjection[viewport] * TransformToLocalSpace(gridCoordinates.x, gridCoordinates.y, disparity);

  vec2 textureCoordinates = vec2(disparitySampleCoordinates) * texCoordStep;
  v2f.texCoord = textureCoordinates + (vec2(quadCoordOffset) * texCoordStep * pointScale);
}

