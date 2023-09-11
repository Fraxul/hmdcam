#version 310 es
in vec3 position;
out vec4 fragColor;

layout(std140) uniform FrustumVisualizeUniformBlock {
  mat4 viewProjection;
  mat4 frustumViewProjectionInverse;
  vec4 color;
};

void main() {
  gl_Position = viewProjection * (frustumViewProjectionInverse * vec4(position, 1.0));
  fragColor = color;
}

