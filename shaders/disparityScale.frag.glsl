#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform highp isampler2D imageTex;

layout(std140) uniform DisparityScaleUniformBlock {
  float disparityScale;
  float pad2;
  float pad3;
  float pad4;
};

void main() {
  float disparity = float(abs(texelFetch(imageTex, ivec2(gl_FragCoord.xy), 0).r));
  outColor = vec4(vec3(disparity * disparityScale), 1.0f);

  // convert to luma
}


