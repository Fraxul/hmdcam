#version 320 es
in vec2 fragTexCoord;
out vec4 outColor;

uniform sampler2D sTexture;

void main() {
   outColor = texture(sTexture, fragTexCoord);
}

