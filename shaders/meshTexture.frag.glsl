#version 320 es

in vec2 v2fTexCoord;
in vec4 v2fPosition;
flat in int v2fTrimmed;
uniform SAMPLER_TYPE imageTex;
uniform sampler2D distortionMap;

/*
uniform vec3 tolforconf;
layout(location = 7) uniform vec4 coloruniform;
*/
layout(location = 0) out vec4 outColor;

void main()
{
/*
	float forcefloorconf = 1.0;
	if( posout.y < tolforconf.z ) forcefloorconf = (posout.y - tolforconf.z)*10.0+ 1.0;
	outputColor = vec4( pow( texture(mytexture, v2UVcoords).rgb * vec3(1.0,1.0,1.0), vec3(1.0) ), 
		clamp(posout.w * tolforconf.x + tolforconf.y, 0.0, 1.0)*forcefloorconf ) * coloruniform;
*/

//  if (v2fPosition.w < 0.1)
//    discard;

  if (v2fTrimmed != 0)
    discard;

  //outColor = texture(imageTex, v2fTexCoord);

  // Remap through OpenCV-generated distortion map
  vec2 distortionCoord = texture(distortionMap, v2fTexCoord).rg; // RG32F texture
  outColor = texture(imageTex, distortionCoord);
}

