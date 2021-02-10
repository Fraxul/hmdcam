#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform sampler2D imageTex;

layout(std140) uniform DisparityScaleUniformBlock {
  float disparityScale;
  float pad2;
  float pad3;
  float pad4;
};

void main() {
  float disparity = texture(imageTex, fragTexCoord).r;
  outColor = vec4(vec3(disparity * disparityScale), 1.0f);

  // convert to luma
}


