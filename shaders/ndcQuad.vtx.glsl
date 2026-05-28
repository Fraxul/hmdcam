#version 310 es
in vec4 position;
in vec2 textureCoordinates;
out vec2 fragTexCoord;

void main() {
  // No transform -- this shader is intended for fullscreen passes.
  gl_Position = position;
  fragTexCoord = textureCoordinates;
}

