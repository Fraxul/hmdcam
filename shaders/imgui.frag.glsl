#version 320 es

in vec2 Frag_UV;
in vec4 Frag_Color;
uniform sampler2D sTexture;
layout (location = 0) out vec4 Out_Color;
void main() {
  Out_Color = Frag_Color * texture(sTexture, Frag_UV);
}

