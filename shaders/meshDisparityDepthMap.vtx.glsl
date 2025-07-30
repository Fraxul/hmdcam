#version 310 es

#include "MeshDisparityDepthMapUniformBlock.h"
#include "MeshDisparityDepthMapCommon.h"

// xy is image texture coordinates (0...1)
// zw is disparity map coordinates (integer texels)
layout(location = 0) in vec4 textureCoordinates;

#if DISPARITY_USE_FP16
uniform sampler2D disparityTex;
float sampleDisparity(ivec2 mipCoords) {
  return texelFetch(disparityTex, mipCoords, 0).r;
}
#else
uniform highp usampler2D disparityTex;
float sampleDisparity(ivec2 mipCoords) {
  return float(texelFetch(disparityTex, mipCoords, 0).r);
}
#endif

out V2G {
  vec4 P;
  vec2 texCoord;
  flat int trimmed;
} v2g;

void main()
{
  float disparityRaw = 0.0f;
  if (debugFixedDisparity >= 0) {
    disparityRaw = float(debugFixedDisparity);
  } else {
    disparityRaw = sampleDisparity(ivec2(textureCoordinates.zw));
  }
  
  if (disparityRaw > float(maxValidDisparityRaw)) {
    // Invalid disparity, so we discard this point.
    v2g.trimmed = 1;
  } else {
    v2g.trimmed = int(any(notEqual(clamp(textureCoordinates.zw, trim_minXY, trim_maxXY), textureCoordinates.zw)));
  }

  float disparity = max(disparityRaw * disparityPrescale, (1.0f / 32.0f)); // prescale and prevent divide-by-zero
  v2g.P = TransformToLocalSpace(textureCoordinates, disparity);
  v2g.texCoord = textureCoordinates.xy;
}

