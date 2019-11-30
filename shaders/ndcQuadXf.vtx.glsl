#version 320 es
in vec4 position;
in vec2 textureCoordinates;
out vec2 fragTexCoord;

layout(std140) uniform NDCQuadUniformBlock {
  mat4 modelViewProjection;
};

void main() {
  gl_Position = modelViewProjection * position;
  fragTexCoord = textureCoordinates;
}


