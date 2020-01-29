// "Borrowed" from monado src/xrt/compositor/shaders/vive.frag @ fca0513b4e8b1920022e1450825e4920cd1a9a17
// Copyright      2017, Philipp Zabel.
// Copyright 2017-2019, Collabora, Ltd.
// SPDX-License-Identifier: BSL-1.0
// Author: Philipp Zabel <philipp.zabel@gmail.com>
// Author: Lubosz Sarnecki <lubosz.sarnecki@collabora.com>

#version 320 es

//per eye texture to warp for lens distortion
uniform sampler2D imageTex;

layout (std140) uniform ViveDistortionUniformBlock {
	vec4 coeffs[3];
	vec4 center;
	float undistort_r2_cutoff;
	float aspect_x_over_y;
	float grow_for_undistort;
  float pad4;
} ubo;

in vec2 fragTexCoord;
layout (location = 0) out vec4 outColor;


void main() {
	vec2 factor = 0.5 / (1.0 + ubo.grow_for_undistort)
	              * vec2(1.0, ubo.aspect_x_over_y);

  // Rescale to -1...1 coordinates
	vec2 texCoord = 2.0 * fragTexCoord - vec2(1.0);

	texCoord.y /= ubo.aspect_x_over_y;
	texCoord -= ubo.center.xy;

  // Center distance squared
	float r2 = dot(texCoord, texCoord);

	vec3 d_inv = ((r2 * ubo.coeffs[2].xyz + ubo.coeffs[1].xyz)
	             * r2 + ubo.coeffs[0].xyz)
	             * r2 + vec3(1.0);

	vec3 d = 1.0 / d_inv;

	const vec2 offset = vec2(0.5);

	vec2 tc_r = offset + (texCoord * d.r + ubo.center.xy) * factor;
	vec2 tc_g = offset + (texCoord * d.g + ubo.center.xy) * factor;
	vec2 tc_b = offset + (texCoord * d.b + ubo.center.xy) * factor;

	vec3 color = vec3(
	      texture(imageTex, tc_r).r,
	      texture(imageTex, tc_g).g,
	      texture(imageTex, tc_b).b);

#if 1
	if (r2 > ubo.undistort_r2_cutoff) {
		color *= 0.125;
	}
#endif

	outColor = vec4(color, 1.0);
}
