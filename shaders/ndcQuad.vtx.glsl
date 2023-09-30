#version 310 es
in vec4 position;
in vec2 textureCoordinates;
out vec2 fragTexCoord;

void main() {
  gl_Position = position;
  fragTexCoord = textureCoordinates;
}

