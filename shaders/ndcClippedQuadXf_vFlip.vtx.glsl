#version 310 es
// in vec4 position; // Unused, recreated from textureCoordinates
in vec2 textureCoordinates;
out vec2 fragTexCoord;

layout(std140) uniform NDCClippedQuadUniformBlock {
  mat4 modelViewProjection;
  vec2 minUV;
  vec2 maxUV;
};

void main() {
  vec2 clippedTexcoord = mix(minUV, maxUV, textureCoordinates);
  vec4 clippedPosition = vec4((clippedTexcoord * vec2(2.0f)) - vec2(1.0f), 0.0f, 1.0f);
  gl_Position = modelViewProjection * clippedPosition;
  fragTexCoord = flipTexcoordY(clippedTexcoord);
}

