#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform SAMPLER_TYPE imageTex;
uniform sampler2D distortionMap;
void main() {
  // Remap through OpenCV-generated distortion map
  vec2 distortionCoord = texture(distortionMap, fragTexCoord).rg; // RG32F texture
  vec3 color = texture(imageTex, distortionCoord).rgb;
  // convert to luma
  outColor = vec4(vec3(dot(color, vec3(0.212671f, 0.715160f, 0.072169f))), 1.0f);
}


