layout(std140) uniform MeshDisparityDepthMapUniformBlock {
  mat4 modelViewProjection[2];
  mat4 R1;
  mat4 colorReprojection; // IMU depth-timewarp homography on the rectified texcoord (identity when disabled).
  vec4 depthParameters;
  vec2 mogrify;
  float disparityPrescale;
  int debugFixedDisparity;

  vec2 trim_minXY;
  vec2 trim_maxXY;

  int renderStereo;
  float maxValidDisparityPixels;
  uint maxValidDisparityRaw;
  float unused1;

  vec2 texCoordStep; // (1/internalWidth, 1/internalHeight)
  float minDepthCutoff;
  float pointScale;

  vec2 inputImageSize;
  float viewZFightBiasMeters; // metric depth bias toward the camera for the DEPTH_BIAS variant
  float pad4;

  vec4 projectionColumn2[2]; // per-eye projection matrix 3rd column, for eye-space-Z depth biasing
};
