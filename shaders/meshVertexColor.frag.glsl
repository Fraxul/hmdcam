#version 310 es
precision highp float;

in vec4 v2fColor;

layout(location = 0) out vec4 outColor;

void main() {
  outColor = v2fColor;
}

