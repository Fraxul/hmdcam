#version 310 es
precision highp float;
in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;
uniform samplerExternalOES imageTex;
uniform sampler2D distortionMap;
uniform sampler2D maskTex;
void main() {
  // Remap through OpenCV-generated distortion map
  vec2 distortionCoord = texture(distortionMap, fragTexCoord).rg; // RG32F texture

//  outColor = vec4(distortionCoord, 0.0, 1.0);
//  outColor = vec4(fragTexCoord, 0.0, 1.0);
//  gl_FragColor = texture2D(imageTex, fragTexCoord);

  if (any(equal(distortionCoord.xyxy, vec4(0.0f, 0.0f, 1.0f, 1.0f)))) { // clip edge pixels
    outColor = vec4(0.0);
  } else {
    outColor = texture(imageTex, distortionCoord) * texture(maskTex, distortionCoord).r;
  }

}


