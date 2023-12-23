#version 310 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform SAMPLER_TYPE imageTex;
void main() {
  outColor = texture(imageTex, fragTexCoord);
}

