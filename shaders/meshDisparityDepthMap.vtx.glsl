#version 320 es

layout(std140) uniform MeshDisparityDepthMapUniformBlock {
  mat4 modelViewProjection[2];
  mat4 R1inv;
  float Q3, Q7, Q11;
  float CameraDistanceMeters;
  vec2 mogrify;
  float disparityPrescale;
  int disparityTexLevels;

  vec2 trim_minXY;
  vec2 trim_maxXY;

  int renderStereo;
  float maxValidDisparityPixels;
  int maxValidDisparityRaw;
  float maxDepthDiscontinuity;
};

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 textureCoordinates;

uniform highp isampler2D disparityTex;

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
  int disparityRaw = 0;
  ivec2 mipCoords = ivec2(textureCoordinates.zw);
  // Walk the mip chain to find a valid disparity value at this location
  for (int level = 0; level < disparityTexLevels; ++level) {
    disparityRaw = texelFetch(disparityTex, mipCoords, level).r;
    if (disparityRaw > 0 && disparityRaw < maxValidDisparityRaw)
      break;
    mipCoords = mipCoords >> 1;
  }
  
  disparityRaw = max(disparityRaw, 1); // prevent divide-by-zero

  float disparity = (float(abs(disparityRaw)) * disparityPrescale);
  v2g.P = TransformToLocalSpace(textureCoordinates.z, textureCoordinates.w, disparity);
  v2g.texCoord = textureCoordinates.xy;
  v2g.trimmed = int(any(notEqual(clamp(textureCoordinates.zw, trim_minXY, trim_maxXY), textureCoordinates.zw)));
}

