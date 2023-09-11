#version 310 es
in vec4 position;
out vec4 fragColor;

layout(std140) uniform SolidQuadUniformBlock {
  mat4 modelViewProjection;
  vec4 color;
};

void main() {
  gl_Position = modelViewProjection * position;
  fragColor = color;
}


