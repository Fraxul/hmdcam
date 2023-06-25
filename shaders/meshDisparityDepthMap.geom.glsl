#version 320 es
layout(triangles) in;
layout(triangle_strip, max_vertices=6) out;

#include "MeshDisparityDepthMapUniformBlock.h"

in V2G {
  vec4 P;
  vec2 texCoord;
  flat int trimmed;
} v2g[];

out G2F {
  vec2 texCoord;
} g2f;

// No output block declaration required.
void main() {
  if (v2g[0].trimmed != 0)
    return;

  if (max(max(abs(v2g[0].P.z), abs(v2g[1].P.z)), abs(v2g[2].P.z)) < minDepthCutoff)
    return;

  float discontinuity = max(abs(v2g[2].P.z - v2g[0].P.z), abs(v2g[1].P.z - v2g[0].P.z));
  if (discontinuity > maxDepthDiscontinuity)
    return;

  for (int viewport = 0; viewport < 2; ++viewport) {
    gl_ViewportIndex = viewport;
    for (int vtx = 0; vtx < 3; ++vtx) {
      gl_Position = modelViewProjection[viewport] * v2g[vtx].P;
      g2f.texCoord = v2g[vtx].texCoord;
      EmitVertex();
    }
    EndPrimitive();
    if (renderStereo == 0) {
      return;
    }
  }
}

