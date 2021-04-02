#version 320 es

layout(std140) uniform MeshDisparityDepthMapUniformBlock {
  mat4 modelViewProjection;
  mat4 R1inv;
  float Q3, Q7, Q11;
  float CameraDistanceMeters;
  vec2 mogrify;
  float disparityPrescale;
  int disparityTexLevels;
};

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 textureCoordinates;

uniform highp isampler2D disparityTex;

out vec2 v2fTexCoord;
out vec4 v2fPosition;

vec4 TransformToLocalSpace( float x, float y, float fDisp ) {

	float lz = Q11 * CameraDistanceMeters / (fDisp * mogrify.x);
	float ly = -((y * mogrify.y) + Q7) / Q11;
	float lx = ((x * mogrify.x) + Q3) / Q11;
	lx *= lz;
	ly *= lz;
	lz *= -1.0f;
	return R1inv * vec4(lx, ly, lz, 1.0f);
}

void main()
{
	v2fPosition = position;
	v2fTexCoord = textureCoordinates.xy;

  int disparityRaw;
  ivec2 mipCoords = ivec2(textureCoordinates.zw);
  // Walk the mip chain to find a valid disparity value at this location
  for (int level = 0; level < disparityTexLevels; ++level) {
    disparityRaw = texelFetch(disparityTex, mipCoords, level).r;
    if (disparityRaw > 0)
      break;
    mipCoords = mipCoords >> 1;
  }
  
  disparityRaw = max(disparityRaw, 1); // prevent divide-by-zero

  float disparity = (float(abs(disparityRaw)) * disparityPrescale);
  vec4 p = TransformToLocalSpace(textureCoordinates.z, textureCoordinates.w, disparity);

	gl_Position = modelViewProjection * p;
}
