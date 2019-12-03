#version 320 es
// Definitions for commonDeclarations.glsl functions

#ifdef CAMERA_INVERTED // Should be defined when this function is used
vec2 adjustCameraTexcoord(vec2 texCoord) {
  if (CAMERA_INVERTED != 0) {
    // Convert origin convention and rotate image 180 degrees
    return vec2(1.0f - texCoord.x, texCoord.y);
  } else {
    // Convert from left-top origin used by libargus to OpenGL left-bottom origin convention
    return vec2(texCoord.x, 1.0f - texCoord.y);
  }
}
#endif

vec2 flipTexcoordY(vec2 texCoord) {
  return vec2(texCoord.x, 1.0f - texCoord.y);
}

