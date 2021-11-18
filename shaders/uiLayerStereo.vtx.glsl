#version 320 es
in vec3 position;
in vec2 textureCoordinates;

out V2G {
  vec4 P;
  vec2 texCoord;
} v2g;

void main() {
  v2g.P = vec4(position, 1.0);
  v2g.texCoord = textureCoordinates;
}

