#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform samplerExternalOES imageTex;
uniform sampler2D overlayTex;
void main() {
  vec4 color = texture(imageTex, fragTexCoord);
  // Flip to convert overlay texture's OpenCV coordinate system to GL
  vec4 overlayColor = texture(overlayTex, vec2(fragTexCoord.x, 1.0f - fragTexCoord.y));

  // OpenCV's overlay drawing doesn't set the alpha channel, so we just assume that
  // the pixels are opaque if they're not black.
  overlayColor.a = ceil(dot(overlayColor.rgb, vec3(1.0f)));
  outColor = (color * (1.0f - overlayColor.a)) + (overlayColor * overlayColor.a);
}


