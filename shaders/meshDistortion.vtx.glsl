#version 320 es
in vec4 position_Ruv;
out vec2 Ruv;

#if CHROMA_CORRECTION
in vec4 Guv_Buv;
out vec2 Guv;
out vec2 Buv;
#endif


void main() {
  gl_Position = vec4(position_Ruv.xy, 0.0f, 1.0f);
  Ruv = position_Ruv.zw;
#if CHROMA_CORRECTION
  Guv = Guv_Buv.xy;
  Buv = Guv_Buv.zw;
#endif
}


