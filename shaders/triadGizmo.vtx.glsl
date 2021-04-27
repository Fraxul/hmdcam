#version 320 es

#if 0
layout(std140) buffer MatrixDataBuffer {
  mat4 instanceMatrices[];
};
#else
layout(std140) buffer PositionDataBuffer {
  vec4 instancePositions[];
};
#endif

layout(std140) uniform TriadGizmoUniformBuffer {
  mat4 viewProjection;
};

in vec3 position;
in vec3 color;

out vec4 fragColor;

void main() {
  //gl_Position = viewProjection * instanceMatrices[gl_InstanceID] * vec4(position, 1.0f); // matrix version
  gl_Position = viewProjection * vec4(instancePositions[gl_InstanceID].xyz + position.xyz, 1.0f); // point version
  fragColor = vec4(color, 1.0f);
}

