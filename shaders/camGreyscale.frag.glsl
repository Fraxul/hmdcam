#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform samplerExternalOES imageTex;
void main() {
  vec3 color = texture(imageTex, fragTexCoord).rgb;
  // convert to luma
  outColor = vec4(vec3(dot(color, vec3(0.212671f, 0.715160f, 0.072169f))), 1.0f);
}

