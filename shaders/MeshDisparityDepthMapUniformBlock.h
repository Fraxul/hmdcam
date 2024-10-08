layout(std140) uniform MeshDisparityDepthMapUniformBlock {
  mat4 modelViewProjection[2];
  mat4 R1inv;
  float Q3, Q7, Q11;
  float CameraDistanceMeters;
  vec2 mogrify;
  float disparityPrescale;
  int disparityTexLevels;

  vec2 trim_minXY;
  vec2 trim_maxXY;

  int renderStereo;
  float maxValidDisparityPixels;
  uint maxValidDisparityRaw;
  float maxDepthDiscontinuity;

  vec2 texCoordStep; // (1/internalWidth, 1/internalHeight)
  float minDepthCutoff;
  float pointScale;

  int debugFixedDisparity;
  float pad2, pad3, pad4;
};


