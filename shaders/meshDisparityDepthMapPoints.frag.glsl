#version 310 es

in V2F {
  vec2 texCoord;
  flat int trimmed;
} v2f;
uniform SAMPLER_TYPE imageTex;

layout(location = 0) out vec4 outColor;

void main()
{
  if (v2f.trimmed > 0)
    discard;

  outColor = texture(imageTex, v2f.texCoord);
}


