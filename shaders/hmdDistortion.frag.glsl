#version 320 es
// Modified from the shader string embedded in OpenHMD, typically returned
// when calling ohmd_gets(OHMD_GLSL_ES_DISTORTION_FRAG_SRC, &fragment);

in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

//per eye texture to warp for lens distortion
uniform sampler2D imageTex;

layout(std140) uniform HMDDistortionUniformBlock {
  //Distoriton coefficients (PanoTools model) [a,b,c,d]
  uniform vec4 HmdWarpParam;

  //chromatic distortion post scaling
  uniform vec3 aberr;

  //Position of lens center in m (usually eye_w/2, eye_h/2)
  uniform vec2 LensCenter;
  //Scale from texture co-ords to m (usually eye_w, eye_h)
  uniform vec2 ViewportScale;
  //Distortion overall scale in m (usually ~eye_w/2)
  uniform float WarpScale;
};

void main() {
  //output_loc is the fragment location on screen from [0,1]x[0,1]
  vec2 output_loc = vec2(fragTexCoord.s, fragTexCoord.t);
  //Compute fragment location in lens-centered co-ordinates at world scale
  vec2 r = output_loc * ViewportScale - LensCenter;
  //scale for distortion model
  //distortion model has r=1 being the largest circle inscribed (e.g. eye_w/2)
  r /= WarpScale;

  //|r|**2
  float r_mag = length(r);
  //offset for which fragment is sourced
  vec2 r_displaced = r * (HmdWarpParam.w + HmdWarpParam.z * r_mag +
  HmdWarpParam.y * r_mag * r_mag +
  HmdWarpParam.x * r_mag * r_mag * r_mag);
  //back to world scale
  r_displaced *= WarpScale;
  //back to viewport co-ord
  vec2 tc_r = (LensCenter + aberr.r * r_displaced) / ViewportScale;
  vec2 tc_g = (LensCenter + aberr.g * r_displaced) / ViewportScale;
  vec2 tc_b = (LensCenter + aberr.b * r_displaced) / ViewportScale;

  float red = texture(imageTex, tc_r).r;
  float green = texture(imageTex, tc_g).g;
  float blue = texture(imageTex, tc_b).b;
  //Black edges off the texture
  outColor = ((tc_g.x < 0.0) || (tc_g.x > 1.0) || (tc_g.y < 0.0) || (tc_g.y > 1.0)) ? vec4(0.0, 0.0, 0.0, 1.0) : vec4(red, green, blue, 1.0);
}

