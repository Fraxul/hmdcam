#version 320 es
layout(triangles) in;
layout(triangle_strip, max_vertices=6) out;

layout(std140) uniform UILayerStereoUniformBlock {
  mat4 modelViewProjection[2];
};

in V2G {
  vec4 P;
  vec2 texCoord;
} v2g[];

out G2F {
  vec2 texCoord;
} g2f;

// No output block declaration required.
void main() {
  for (int viewport = 0; viewport < 2; ++viewport) {
    gl_ViewportIndex = viewport;
    for (int vtx = 0; vtx < 3; ++vtx) {
      gl_Position = modelViewProjection[viewport] * v2g[vtx].P;
      g2f.texCoord = v2g[vtx].texCoord;
      EmitVertex();
    }
    EndPrimitive();
  }
}

