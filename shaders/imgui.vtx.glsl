#version 310 es
in vec2 Position;
in vec2 UV;
in vec4 Color;

out vec2 Frag_UV;
out vec4 Frag_Color;

layout(std140) uniform UILayerUniformBlock {
  mat4 modelViewProjection;
};

void main() {
  gl_Position = modelViewProjection * vec4(Position.xy, 0.0f, 1.0f);
  // gl_ViewportIndex = gl_InstanceID; // Viewport index
  Frag_UV = UV;
  Frag_Color = Color;
}


