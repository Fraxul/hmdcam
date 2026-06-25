#version 310 es
out vec4 outColor;

void main() {
  // Debug visualization: paint the masked border red so the hidden-area mesh is
  // visible on the eye target. In production this pass should be depth-only
  // (bind a pipeline with color writes disabled) to save bandwidth -- the depth
  // write in the vertex shader is what actually drives the optimization.
  outColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
}
