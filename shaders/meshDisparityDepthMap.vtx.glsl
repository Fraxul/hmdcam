#version 310 es

#include "MeshDisparityDepthMapUniformBlock.h"

// xy is image texture coordinates (0...1)
// zw is disparity map coordinates (integer texels)
layout(location = 0) in vec4 textureCoordinates;

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
  float disparityRaw = 0.0f;
  if (debugFixedDisparity >= 0) {
    disparityRaw = float(debugFixedDisparity);

  } else {
    ivec2 mipCoords = ivec2(textureCoordinates.zw);
    // Walk the mip chain to find a valid disparity value at this location
    for (int level = 0; level < disparityTexLevels; ++level) {
      disparityRaw = sampleDisparity(mipCoords, level);
      if (disparityRaw <= float(maxValidDisparityRaw))
        break;
      mipCoords = mipCoords >> 1;
    }
  }
  

  float disparity = max(disparityRaw * disparityPrescale, (1.0f / 32.0f)); // prescale and prevent divide-by-zero
  v2g.P = TransformToLocalSpace(textureCoordinates.z, textureCoordinates.w, disparity);
  v2g.texCoord = textureCoordinates.xy;
  v2g.trimmed = int(any(notEqual(clamp(textureCoordinates.zw, trim_minXY, trim_maxXY), textureCoordinates.zw)));
}

