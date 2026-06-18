#version 310 es

#if DEPTH_BIAS
#extension GL_EXT_conservative_depth : require // enables layout(depth_greater) on gl_FragDepth
#endif

in V2F {
  vec2 texCoord;
#if DEPTH_BIAS
  vec4 biasedClipPos;
#endif
} v2f;
uniform sampler2D imageTex;
uniform sampler2D distortionMap;

layout(location = 0) out vec4 outColor;

#if DEPTH_BIAS
// Conservative depth: the toward-camera bias can only increase the written depth above
// gl_FragCoord.z under reverse-Z, so depth_greater keeps hierarchical-Z rejection intact.
layout(depth_greater) out highp float gl_FragDepth;
#endif

void main()
{
  // Remap through OpenCV-generated distortion map
  vec2 distortionCoord = texture(distortionMap, v2f.texCoord).rg; // RG32F texture
  outColor = SAMPLE_CAMERA(distortionCoord);

#if DEPTH_BIAS
  // Vulkan reverse-Z: window depth = clip.z / clip.w. Dividing the perspective-correctly-
  // interpolated clip z and w reproduces the exact screen-linear depth of the biased surface.
  //
  // Guard the divide: once the eye-space bias pushes this surface to within viewZFightBiasMeters
  // of the camera, biasedClipPos.w crosses zero and z/w stops being a depth in [0,1] -- it flips
  // sign and, naively clamped, lands on the FAR plane (0.0) so the fragment fails the depth test
  // and vanishes. Pin to the near plane (1.0) whenever the biased point reaches or crosses it.
  // Both branches yield >= gl_FragCoord.z, so the depth_greater promise above still holds.
  gl_FragDepth = (v2f.biasedClipPos.w > 0.0)
      ? min(v2f.biasedClipPos.z / v2f.biasedClipPos.w, 1.0)
      : 1.0;
#endif
}


