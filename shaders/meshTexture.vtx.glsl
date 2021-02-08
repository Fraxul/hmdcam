#version 320 es

layout(std140) uniform MeshTransformUniformBlock {
  mat4 modelViewProjection;
};

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 textureCoordinates;

out vec2 v2fTexCoord;
out vec4 v2fPosition;
void main()
{
	v2fPosition = position;
	v2fTexCoord = textureCoordinates;
	gl_Position = modelViewProjection * vec4( position.xyz, 1.0 );
}
