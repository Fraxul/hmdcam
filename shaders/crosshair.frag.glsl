#version 310 es
#include "CrosshairUniformBlock.h"

in vec2 ndc; // Varies -1...1 across the quad
layout(location = 0) out vec4 outColor;

void main() {

  float d2 = dot(ndc, ndc);
  if (d2 > 1.0 || d2 < (1.0f - thickness)) {
    discard;
  }

  outColor = color;
}

