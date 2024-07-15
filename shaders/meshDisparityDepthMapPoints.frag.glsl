#version 310 es

in V2F {
  vec2 texCoord;
} v2f;
uniform SAMPLER_TYPE imageTex;
uniform sampler2D distortionMap;

layout(location = 0) out vec4 outColor;

void main()
{
  // Remap through OpenCV-generated distortion map
  vec2 distortionCoord = texture(distortionMap, v2f.texCoord).rg; // RG32F texture
  outColor = texture(imageTex, distortionCoord);
}


