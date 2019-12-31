#version 320 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform samplerExternalOES leftCameraTex;
uniform samplerExternalOES rightCameraTex;
uniform sampler2D leftOverlayTex;
uniform sampler2D rightOverlayTex;
void main() {
  vec4 color, overlayColor;
  vec2 viewTexCoord = vec2(fract(fragTexCoord.x * 2.0f), fragTexCoord.y);
  vec2 cameraTexCoord = adjustCameraTexcoord(viewTexCoord);

  if (floor(fragTexCoord.x * 2.0f) == 0.0f) {
    color = texture(leftCameraTex, cameraTexCoord);
    // Flip to convert from overlay texture's OpenCV coordinate system to GL
    overlayColor = texture(leftOverlayTex, flipTexcoordY(viewTexCoord));
  } else {
    color = texture(rightCameraTex, cameraTexCoord);
    // Flip to convert from overlay texture's OpenCV coordinate system to GL
    overlayColor = texture(rightOverlayTex, flipTexcoordY(viewTexCoord));
  }

  // OpenCV's overlay drawing doesn't set the alpha channel, so we just assume that
  // the pixels are opaque if they're not black.
  overlayColor.a = ceil(dot(overlayColor.rgb, vec3(1.0f)));
  outColor = (color * (1.0f - overlayColor.a)) + (overlayColor * overlayColor.a);
}

