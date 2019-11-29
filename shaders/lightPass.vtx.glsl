#version 320 es
in vec3 position;
in vec2 textureCoordinates;

out vec2 fragTexCoord;

#ifndef QUAD_Z_PLANE
#define QUAD_Z_PLANE 1.0f
#endif

void main() {
  // no transform necessary, this is just to render a fullscreen quad.
  gl_Position = vec4(position.xy, QUAD_Z_PLANE, 1.0f);
  fragTexCoord = textureCoordinates;
}

