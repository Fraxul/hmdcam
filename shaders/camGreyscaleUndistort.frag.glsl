#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform samplerExternalOES imageTex;
uniform sampler2D distortionMap;
void main() {
  // Remap through OpenCV-generated distortion map
  vec2 distortionCoord = flipTexcoordY(texture(distortionMap, flipTexcoordY(fragTexCoord)).rg); // RG32F texture

  // Flip coordinate system to match OpenCV's on the output
  vec3 color = texture(imageTex, adjustCameraTexcoord(distortionCoord)).rgb;
  // convert to luma
  outColor = vec4(vec3(dot(color, vec3(0.212671f, 0.715160f, 0.072169f))), 1.0f);
}


