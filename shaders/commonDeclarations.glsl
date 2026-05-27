// This file is injected into each shader -- after the version and preprocessor defines, before the shader body.
//
// Backends gate extensions / declarations via:
//   RHI_TARGET_VULKAN -- defined by the Vulkan RHI backend. When set, all
//     GL-specific extension directives are skipped (Vulkan SPIR-V provides
//     gl_ViewportIndex, image formats, etc. as core features). YUV camera
//     samples used to need samplerExternalOES on GL; on VK they go through
//     VkSamplerYcbcrConversion at sampler creation time (Phase 6) so the
//     shader-side sampler is just plain sampler2D.
//   GL_ES -- predefined by GLSL ES compilers; the legacy GL-on-ES extensions
//     live in this branch.

#ifndef RHI_TARGET_VULKAN
  #ifdef GL_ES
    #extension GL_OES_shader_io_blocks : enable
    #extension GL_OES_geometry_shader : enable
  #endif
  #if RHI_VERTEX_SHADER
    #if defined(GL_ARB_shader_viewport_layer_array) && GL_ARB_shader_viewport_layer_array
      #extension GL_ARB_shader_viewport_layer_array : enable
    #elif defined(GL_NV_viewport_array2) && GL_NV_viewport_array2
      #extension GL_NV_viewport_array2 : enable
    #elif defined(GL_AMD_vertex_shader_viewport_index) && GL_AMD_vertex_shader_viewport_index
      #extension GL_AMD_vertex_shader_viewport_index : enable
    #else
      #define SKIP_VIEWPORT_WRITE
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
    #else
      #extension GL_ARB_viewport_array : require
    #endif
  #elif RHI_COMPUTE_SHADER
    #extension GL_NV_image_formats : require
  #endif
#else // RHI_TARGET_VULKAN
  // Output blocks: core in SPIR-V, but the source profile (ES) needs the
  // extension to accept them.
  #extension GL_EXT_shader_io_blocks : enable
  // Vulkan SPIR-V can write gl_ViewportIndex / gl_Layer from any stage
  // (VK_EXT_shader_viewport_index_layer is widely supported). Enable for
  // all stages, not just vertex.
  #extension GL_ARB_shader_viewport_layer_array : enable
  // Geometry shaders themselves require GL_EXT_geometry_shader in ES <3.20.
  #if RHI_GEOMETRY_SHADER
    #extension GL_EXT_geometry_shader : require
  #endif
  // Built-in renames: Vulkan SPIR-V uses gl_InstanceIndex / gl_VertexIndex
  // (which include the baseInstance / baseVertex). Source still says
  // gl_InstanceID / gl_VertexID, so alias.
  #define gl_InstanceID gl_InstanceIndex
  #define gl_VertexID gl_VertexIndex
#endif // RHI_TARGET_VULKAN

precision highp float;
precision highp int;

vec2 flipTexcoordY(vec2 texCoord);

// Camera-texture X-coord crop: when the camera VkImage was allocated with
// extent.width > valid Argus width (NvBuf pitch-padding workaround on Tegra),
// callers must scale their normalized X coord into the valid sub-region
// before sampling. The host sets CAM_TEX_CROP_X via setFlag() to validW /
// extentW when needed (e.g. 1920/2048 = 0.9375); default 1.0 means no crop.
#ifndef CAM_TEX_CROP_X
#define CAM_TEX_CROP_X 1.0
#endif
#define SAMPLE_CAMERA(uv) texture(imageTex, vec2((uv).x * CAM_TEX_CROP_X, (uv).y))

