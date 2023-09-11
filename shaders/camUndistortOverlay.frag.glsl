#version 310 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform samplerExternalOES imageTex;
uniform sampler2D distortionMap;
uniform sampler2D overlayTex;
void main() {
  // Remap through OpenCV-generated distortion map
  vec2 distortionCoord = texture(distortionMap, fragTexCoord).rg; // RG32F texture
  vec4 overlayColor = texture(overlayTex, fragTexCoord);

  // OpenCV's overlay drawing doesn't set the alpha channel, so we just assume that
  // the pixels are opaque if they're not black.
  overlayColor.a = ceil(dot(overlayColor.rgb, vec3(1.0f)));

  vec4 cameraColor;
  if (any(equal(distortionCoord.xyxy, vec4(0.0f, 0.0f, 1.0f, 1.0f)))) { // clip edge pixels
    cameraColor = vec4(0.0);
  } else {
    cameraColor = texture(imageTex, distortionCoord);
  }

  outColor = (cameraColor * (1.0f - overlayColor.a)) + (overlayColor * overlayColor.a);
}


