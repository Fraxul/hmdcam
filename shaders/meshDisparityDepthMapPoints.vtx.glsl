#version 310 es

#include "MeshDisparityDepthMapUniformBlock.h"


layout(location = 0) in vec2 textureCoordinates;// image texture coordinates (0...1)
layout(location = 1) in vec2 gridCoordinates; // integer texels, varies over the quad
layout(location = 2) in vec2 disparitySampleCoordinates; // integer texels, fixed to the left-top value

uniform highp usampler2D disparityTex;

out V2F {
  vec2 texCoord;
  flat int trimmed;
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
  uint disparityRaw = 0u;
  if (debugFixedDisparity >= 0) {
    disparityRaw = uint(max(debugFixedDisparity, 1)); // prevent divide-by-zero

  } else {
    ivec2 mipCoords = ivec2(disparitySampleCoordinates);
    // Walk the mip chain to find a valid disparity value at this location
    for (int level = 0; level < disparityTexLevels; ++level) {
      disparityRaw = texelFetch(disparityTex, mipCoords, level).r;
      if (disparityRaw < maxValidDisparityRaw)
        break;
      mipCoords = mipCoords >> 1;
    }

    disparityRaw = max(disparityRaw, 1u); // prevent divide-by-zero
  }

  int viewport = gl_InstanceID;
  float disparity = (float(disparityRaw) * disparityPrescale);
  gl_Position = modelViewProjection[viewport] * TransformToLocalSpace(gridCoordinates.x, gridCoordinates.y, disparity);
  gl_ViewportIndex = viewport;

  v2f.texCoord = textureCoordinates;
  v2f.trimmed = int(any(notEqual(clamp(disparitySampleCoordinates.xy, trim_minXY, trim_maxXY), disparitySampleCoordinates.xy)));
}

