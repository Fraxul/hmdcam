#version 320 es

#include "MeshDisparityDepthMapUniformBlock.h"

// xy is image texture coordinates (0...1)
// zw is disparity map coordinates (integer texels)
layout(location = 0) in vec4 textureCoordinates;

uniform highp usampler2D disparityTex;

out V2G {
  vec4 P;
  vec2 texCoord;
  flat int trimmed;
} v2g;

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
    ivec2 mipCoords = ivec2(textureCoordinates.zw);
    // Walk the mip chain to find a valid disparity value at this location
    for (int level = 0; level < disparityTexLevels; ++level) {
      disparityRaw = texelFetch(disparityTex, mipCoords, level).r;
      if (disparityRaw < maxValidDisparityRaw)
        break;
      mipCoords = mipCoords >> 1;
    }

    disparityRaw = max(disparityRaw, 1u); // prevent divide-by-zero
  }
  

  float disparity = (float(disparityRaw) * disparityPrescale);
  v2g.P = TransformToLocalSpace(textureCoordinates.z, textureCoordinates.w, disparity);
  v2g.texCoord = textureCoordinates.xy;
  v2g.trimmed = int(any(notEqual(clamp(textureCoordinates.zw, trim_minXY, trim_maxXY), textureCoordinates.zw)));
}

