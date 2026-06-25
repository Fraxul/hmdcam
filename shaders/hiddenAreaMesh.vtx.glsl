#version 310 es

// Hidden-area mesh: an annulus covering the region of the eye target that the
// lens distortion pass never samples. Vertices are precomputed in eye-viewport
// NDC on the CPU (see buildHiddenAreaMesh() in Render.cpp).
in vec2 ndcPosition;

void main() {
  // Emit at the reverse-Z near plane (depth 1.0). Combined with the scene's
  // GREATER depth test, this rejects every subsequent fragment in the masked
  // region, since nothing can satisfy depth > 1.0.
  gl_Position = vec4(ndcPosition, 1.0f, 1.0f);
}
