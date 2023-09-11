#version 310 es
// Definitions for commonDeclarations.glsl functions

vec2 flipTexcoordY(vec2 texCoord) {
  return vec2(texCoord.x, 1.0f - texCoord.y);
}

