#version 320 es
in G2F {
  vec2 texCoord;
} g2f;

out vec4 outColor;

uniform sampler2D imageTex;

void main() {
   outColor = texture(imageTex, g2f.texCoord);
}

