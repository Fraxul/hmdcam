#version 320 es
in vec4 position;
in vec2 textureCoordinates;
out vec2 TexCoordOut;

layout(std140) uniform CamTexturedQuadUniformBlock {
  mat4 modelViewProjection;
};

void main() {
  gl_Position = modelViewProjection * position;
  TexCoordOut = textureCoordinates;
}

