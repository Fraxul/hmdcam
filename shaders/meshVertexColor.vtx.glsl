#version 320 es
precision highp float;
in vec4 position;
in vec4 color;

out vec4 v2fColor;
layout(std140) uniform MeshTransformUniformBlock {
  mat4 modelViewProjection;
};

void main() {
  gl_Position = modelViewProjection * position;
  v2fColor = color;
}


