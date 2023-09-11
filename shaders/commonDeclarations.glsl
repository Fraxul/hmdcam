// This file is injected into each shader -- after the version and preprocessor defines, before the shader body.
#ifdef GL_ES
#extension GL_OES_shader_io_blocks : enable
#endif
#if RHI_VERTEX_SHADER
  #if defined(GL_ARB_shader_viewport_layer_array) && GL_ARB_shader_viewport_layer_array
    #extension GL_ARB_shader_viewport_layer_array : enable
  #elif defined(GL_NV_viewport_array2) && GL_NV_viewport_array2
    #extension GL_NV_viewport_array2 : enable
  #elif defined(GL_AMD_vertex_shader_viewport_index) && GL_AMD_vertex_shader_viewport_index
    #extension GL_AMD_vertex_shader_viewport_index : enable
  #endif

#elif RHI_GEOMETRY_SHADER
  #ifdef GL_ES
    #extension GL_OES_viewport_array : require
  #else
    #extension GL_ARB_viewport_array : require
  #endif
#elif RHI_FRAGMENT_SHADER
  #ifdef GL_ES
    #extension GL_OES_viewport_array : require
    #extension GL_OES_EGL_image_external : require
    #extension GL_OES_EGL_image_external_essl3 : require
  #else
    #extension GL_ARB_viewport_array : require
  #endif
#elif RHI_COMPUTE_SHADER
  #extension GL_NV_image_formats : require

#endif

precision highp float;
precision highp int;

vec2 flipTexcoordY(vec2 texCoord);

