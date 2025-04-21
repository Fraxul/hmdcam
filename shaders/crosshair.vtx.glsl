#version 310 es
in vec4 position;
out vec2 ndc;

#include "CrosshairUniformBlock.h"

void main() {
#ifdef SKIP_VIEWPORT_WRITE
  int viewport = 0;
#else
  int viewport = gl_InstanceID;
  gl_ViewportIndex = viewport;
#endif

  gl_Position = modelViewProjection[viewport] * position;
  ndc = position.xy;
}


