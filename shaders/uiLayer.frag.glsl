#version 310 es
in vec2 fragTexCoord;
out vec4 outColor;

uniform sampler2D imageTex;

void main() {
   outColor = texture(imageTex, fragTexCoord);
}

