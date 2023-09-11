#version 310 es
in vec3 position;
in vec2 textureCoordinates;

out vec2 fragTexCoord;

layout(std140) uniform UILayerUniformBlock {
  mat4 modelViewProjection;
};

void main() {
  gl_Position = modelViewProjection * vec4(position, 1.0);
  fragTexCoord = textureCoordinates;
}

