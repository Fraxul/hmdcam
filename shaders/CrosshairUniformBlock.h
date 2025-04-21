layout(std140) uniform CrosshairUniformBlock {
  mat4 modelViewProjection[2];
  vec4 color;
  float thickness;
  float pad2, pad3, pad4;
};

