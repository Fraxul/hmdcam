#version 310 es

layout(location = 0) in V2F {
  vec2 texCoord;
} v2f;
uniform sampler2D imageTex;

layout(location = 0) out vec4 outColor;

void main()
{
  // Texture coordinates have already been distortion-corrected in the vertex shader. This is
  // just a straightforward texture application. The Z-fight bias (DEPTH_BIAS variant) is applied
  // in the vertex shader as a gl_Position.z offset, so this shader writes no gl_FragDepth and
  // early-Z/Hi-Z stays enabled.
  outColor = SAMPLE_CAMERA(v2f.texCoord);
}
