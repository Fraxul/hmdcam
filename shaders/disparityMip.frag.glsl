#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out ivec4 outInt;
uniform highp isampler2D imageTex;

layout(std140) uniform DisparityMipUniformBlock {
  int sourceLevel;
  int maxValidDisparityRaw;
  float pad3, pad4;
};

void main() {
  ivec2 base = ivec2(gl_FragCoord.xy) * 2;

  float accum = 0.0f;
  int samples = 0;

  ivec2 offsets[] = ivec2[](
    ivec2(0, 0),
    ivec2(0, 1),
    ivec2(1, 0),
    ivec2(1, 1)
  );

  for (int i = 0; i < 4; ++i) {
    int d = texelFetchOffset(imageTex, base, sourceLevel, offsets[i]).r;
    if (d > 0 && d < maxValidDisparityRaw) {
      accum += float(d);
      samples += 1;
    }
  }

  float divisors[] = float[](
    1.0f,
    1.0f,
    0.5f,
    1.0f/3.0f,
    0.25f
  );

  if (samples == 0) {
    outInt = ivec4(-1);    
  } else {
    outInt = ivec4(int(round(accum * divisors[samples])));
  }
}

