layout(std140) uniform MeshDistortionUniformBlock {
  vec2 uvOffset;
  vec2 uvScale;
  vec2 apertureCenter; // lens aperture ellipse, in this (combined-eyeTex) UV space
  vec2 apertureInvRadii; // reciprocal half-axes; (0,0) => never cull
};
