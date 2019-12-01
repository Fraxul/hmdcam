#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform samplerExternalOES imageTex;
uniform sampler2D distortionMap;
void main() {
  vec2 distortionCoord = texture(distortionMap, fragTexCoord).rg; // RG32F texture
  outColor = vec4(distortionCoord, 0.0, 1.0);
//  outColor = vec4(fragTexCoord, 0.0, 1.0);
//  gl_FragColor = texture2D(imageTex, fragTexCoord);

  if (any(notEqual(clamp(distortionCoord, vec2(0.0), vec2(1.0)), distortionCoord))) {
    outColor = vec4(0.0);
  } else {
    outColor = texture(imageTex, adjustCameraTexcoord(distortionCoord));
  }

}


