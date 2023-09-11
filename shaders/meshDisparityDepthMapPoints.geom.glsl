#version 310 es
layout(points) in;
layout(triangle_strip, max_vertices=8) out;

#include "MeshDisparityDepthMapUniformBlock.h"

in V2G {
  vec4 P;
  vec4 dPdU;
  vec4 dPdV;

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

  if (abs(v2g[0].P.z) < minDepthCutoff)
    return;

  // Depth discontinuity doesn't apply in point-rendering mode
  //float discontinuity = max(abs(v2g[2].P.z - v2g[0].P.z), abs(v2g[1].P.z - v2g[0].P.z));
  //if (discontinuity > maxDepthDiscontinuity)
  //  return;

  vec2 scaledTexCoordStep = texCoordStep * pointScale;

  for (int viewport = 0; viewport < 2; ++viewport) {
    gl_Position = modelViewProjection[viewport] * v2g[0].P;
    gl_ViewportIndex = viewport;
    g2f.texCoord = v2g[0].texCoord;
    EmitVertex();

    gl_Position = modelViewProjection[viewport] * (v2g[0].P + v2g[0].dPdV);
    gl_ViewportIndex = viewport;
    g2f.texCoord = v2g[0].texCoord + vec2(0.0f, scaledTexCoordStep.y);
    EmitVertex();

    gl_Position = modelViewProjection[viewport] * (v2g[0].P + v2g[0].dPdU);
    gl_ViewportIndex = viewport;
    g2f.texCoord = v2g[0].texCoord + vec2(scaledTexCoordStep.x, 0.0f);
    EmitVertex();

    gl_Position = modelViewProjection[viewport] * (v2g[0].P + v2g[0].dPdU + v2g[0].dPdV);
    gl_ViewportIndex = viewport;
    g2f.texCoord = v2g[0].texCoord + scaledTexCoordStep;
    EmitVertex();

    EndPrimitive();
    if (renderStereo == 0) {
      return;
    }
  }
}

