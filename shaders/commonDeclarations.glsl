// This file is injected into each shader -- after the version and preprocessor defines, before the shader body.
#if RHI_VERTEX_SHADER
  #if defined(GL_ARB_shader_viewport_layer_array) && GL_ARB_shader_viewport_layer_array
    #extension GL_ARB_shader_viewport_layer_array : enable
  #elif defined(GL_NV_viewport_array2) && GL_NV_viewport_array2
    #extension GL_NV_viewport_array2 : enable
  #elif defined(GL_AMD_vertex_shader_viewport_index) && GL_AMD_vertex_shader_viewport_index
    #extension GL_AMD_vertex_shader_viewport_index : enable
  #endif

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

