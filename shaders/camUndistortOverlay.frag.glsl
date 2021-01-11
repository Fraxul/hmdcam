#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform samplerExternalOES imageTex;
uniform sampler2D distortionMap;
uniform sampler2D overlayTex;
void main() {
  // Texture coordinates get flipped for the distortion map lookup (convert from OpenGL -> OpenCV coordsys),
  // then the output gets flipped to convert from OpenCV -> OpenGL coordsys
  vec2 distortionCoord = flipTexcoordY(texture(distortionMap, flipTexcoordY(fragTexCoord)).rg); // RG32F texture

  // Flip to convert from overlay texture's OpenCV coordinate system to GL
  vec4 overlayColor = texture(overlayTex, flipTexcoordY(fragTexCoord));

  // OpenCV's overlay drawing doesn't set the alpha channel, so we just assume that
  // the pixels are opaque if they're not black.
  overlayColor.a = ceil(dot(overlayColor.rgb, vec3(1.0f)));

  vec4 cameraColor;
  if (any(notEqual(clamp(distortionCoord, vec2(0.0), vec2(1.0)), distortionCoord))) {
    cameraColor = vec4(0.0);
  } else {
    cameraColor = texture(imageTex, adjustCameraTexcoord(distortionCoord));
  }

  outColor = (cameraColor * (1.0f - overlayColor.a)) + (overlayColor * overlayColor.a);
}


