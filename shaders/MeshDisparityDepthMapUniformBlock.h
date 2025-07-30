layout(std140) uniform MeshDisparityDepthMapUniformBlock {
  mat4 modelViewProjection[2];
  mat4 R1;
  vec4 depthParameters;
  vec2 mogrify;
  float disparityPrescale;
  int debugFixedDisparity;

  vec2 trim_minXY;
  vec2 trim_maxXY;

  int renderStereo;
  float maxValidDisparityPixels;
  uint maxValidDisparityRaw;
  float maxDepthDiscontinuity;

  vec2 texCoordStep; // (1/internalWidth, 1/internalHeight)
  float minDepthCutoff;
  float pointScale;

  vec2 inputImageSize;
  float pad3, pad4;
};


