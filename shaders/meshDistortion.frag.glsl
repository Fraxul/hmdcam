#version 310 es
#include "MeshDistortionUniformBlock.h"

//per eye texture to warp for lens distortion
uniform sampler2D imageTex;

in vec2 Ruv;
#if CHROMA_CORRECTION
  in vec2 Guv;
  in vec2 Buv;
#endif

layout (location = 0) out vec4 outColor;
void main() {
  // Skip the texture fetch + filtering for panel pixels whose sample lands outside
  // the lens aperture -- these are the vignetted panel corners the eye never sees.
  // The branch is highly coherent (large contiguous corner regions), so warps that
  // are fully outside skip the fetches entirely. apertureInvRadii == 0 disables it.
  vec2 apertureOffset = (Ruv - apertureCenter) * apertureInvRadii;
  if (dot(apertureOffset, apertureOffset) > 1.0f) {
    outColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    return;
  }

  #if CHROMA_CORRECTION
    outColor = vec4(
      texture(imageTex, Ruv).r,
      texture(imageTex, Guv).g,
      texture(imageTex, Buv).b,
      1.0f);
  #else
    outColor = texture(imageTex, Ruv);
  #endif
}

