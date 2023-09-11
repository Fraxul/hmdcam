#version 310 es
precision highp float;

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;

layout(std140) uniform DisparityMipUniformBlock {
  int sourceLevel;
  uint maxValidDisparityRaw;
  float pad3, pad4;
};

layout(binding = 0, r16ui) uniform highp readonly uimage2D srcImage;
layout(binding = 1, r16ui) uniform highp writeonly uimage2D dstMip1;
layout(binding = 2, r16ui) uniform highp writeonly uimage2D dstMip2;
layout(binding = 3, r16ui) uniform highp writeonly uimage2D dstMip3;

shared uint wgDepthSamples[4][4];

uint reduceDepthSamples(uint depthSamples[4]) {
  uint accum = 0u;
  uint validSamples = 0u;

  for (uint i = 0u; i < 4u; ++i) {
    if (depthSamples[i] < maxValidDisparityRaw) {
      accum += depthSamples[i];
      validSamples += 1u;
    }
  }
  return (validSamples == 0u) ? 0xffffu : (accum / validSamples);
}

void main() {
  ivec2 base = ivec2(gl_GlobalInvocationID.xy * 2u);

  uint depthSamples[4];
  ivec2 offsets[] = ivec2[](
    ivec2(0, 0),
    ivec2(0, 1),
    ivec2(1, 0),
    ivec2(1, 1));

  for (int i = 0; i < 4; ++i)
    depthSamples[i] = imageLoad(srcImage, base + offsets[i]).x;

  uint mip0Result = reduceDepthSamples(depthSamples);
  imageStore(dstMip1, ivec2(gl_GlobalInvocationID.xy), uvec4(mip0Result));

  // Exchange results with workgroup members for subsequent mips
  wgDepthSamples[gl_LocalInvocationID.x][gl_LocalInvocationID.y] = mip0Result;

  barrier();

  // Only use the upper 2x2 block (4 threads) for the second mip
  if (gl_LocalInvocationID.x >= 2u || gl_LocalInvocationID.y >= 2u)
    return;

  // Each remaining thread gathers from a 2x2 region from the workgroup-shared intermediary results
  ivec2 mip1Base = ivec2(gl_LocalInvocationID.xy * 2u);
  for (int i = 0; i < 4; ++i) {
    ivec2 coord = mip1Base + offsets[i];
    depthSamples[i] = wgDepthSamples[coord.x][coord.y];
  }
  uint mip1Result = reduceDepthSamples(depthSamples);

  imageStore(dstMip2, ivec2((gl_GlobalInvocationID.xy / 2u) + gl_LocalInvocationID.xy), uvec4(mip1Result));

  // Write results to shared memory for third/final mip
  wgDepthSamples[0][gl_LocalInvocationID.x] = mip1Result;

  barrier();

  // Only use the (0, 0) thread for the third mip

  if (gl_LocalInvocationID.x != 0u)
    return;

  for (int i = 0; i < 4; ++i) {
    depthSamples[i] = wgDepthSamples[0][i];
  }
  uint mip2Result = reduceDepthSamples(depthSamples);
  imageStore(dstMip3, ivec2(gl_GlobalInvocationID.xy / 4u), uvec4(mip2Result));
}

