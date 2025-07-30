vec4 TransformToLocalSpace(vec4 textureCoordinates, float fDisp ) {

  vec3 pp = vec3((textureCoordinates.xy * inputImageSize.xy) + depthParameters.xy, depthParameters.z);
  float lw = depthParameters.w * (fDisp * mogrify.x);
  vec3 res = pp / lw;
  return vec4(mat3(R1) * res, 1.0f);
}


