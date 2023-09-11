#version 310 es
in vec2 fragTexCoord;
out vec4 outColor;

uniform sampler2D sColorTex;

void main() {
   ivec2 fragTexel = ivec2(floor(gl_FragCoord.xy));
   outColor = texelFetch(sColorTex, fragTexel, 0);
}

