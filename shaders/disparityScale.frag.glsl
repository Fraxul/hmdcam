#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform highp isampler2D imageTex;

layout(std140) uniform DisparityScaleUniformBlock {
  float disparityScale;
  int sourceLevel;
  int maxValidDisparityRaw;
  float pad4;
};

void main() {
  ivec2 coord = ivec2(int(gl_FragCoord.x) >> sourceLevel, int(gl_FragCoord.y) >> sourceLevel);

  int disparity_raw = texelFetch(imageTex, coord, sourceLevel).r;
  if (disparity_raw <= 0 || disparity_raw > maxValidDisparityRaw) {
    // Highlight invalid disparity in red
    outColor = vec4(1.0f, 0.0f, 0.0f, 1.0f);
  } else {
    outColor = vec4(vec3(float(disparity_raw) * disparityScale), 1.0f);
  }

  // convert to luma
}


