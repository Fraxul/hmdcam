#version 310 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform sampler2D imageTex;
void main() {
  outColor = SAMPLE_CAMERA(fragTexCoord);
}

