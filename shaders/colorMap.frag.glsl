#version 310 es
precision highp float;
#include "colorMap.h"

in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform highp sampler2D imageTex;

layout(std140) uniform ColorMapUniformBlock {
  float displayRangeMin;
  float displayRangeMax;
  int sourceLevel;
  int colorMapMode;
};


void main() {
  ivec2 coord = ivec2((int(gl_FragCoord.x)) >> sourceLevel, (int(gl_FragCoord.y)) >> sourceLevel);

  float rawValue = texelFetch(imageTex, coord, sourceLevel).r;
  float remappedValue = clamp((rawValue - displayRangeMin) / (displayRangeMax - displayRangeMin), 0.0f, 1.0f);

  vec3 colorMappedValue = applyColormap(remappedValue, colorMapMode);
  outColor = vec4(colorMappedValue, 1.0f);
}

