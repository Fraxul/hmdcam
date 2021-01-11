#version 320 es

//per eye texture to warp for lens distortion
uniform sampler2D imageTex;

in vec2 Ruv;
#if CHROMA_CORRECTION
  in vec2 Guv;
  in vec2 Buv;
#endif

layout (location = 0) out vec4 outColor;
void main() {
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

