// This file is injected into each shader -- after the version and preprocessor defines, before the shader body.
#if RHI_VERTEX_SHADER
#elif RHI_GEOMETRY_SHADER
  #extension GL_OES_viewport_array : require
#elif RHI_FRAGMENT_SHADER
  #extension GL_OES_viewport_array : require
  #extension GL_OES_EGL_image_external : require
  #extension GL_OES_EGL_image_external_essl3 : require

#endif

precision highp float;
precision highp int;

vec2 flipTexcoordY(vec2 texCoord);

