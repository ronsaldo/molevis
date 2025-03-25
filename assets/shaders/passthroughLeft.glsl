#version 450
layout(set=0, binding=0) uniform sampler LinearTextureSampler;
layout(set=1, binding=4) uniform texture2D SourceTexture;

layout(location = 0) in vec2 inTexcoord;
layout(location = 0) out vec4 outColor;

void main()
{
	outColor = textureLod(sampler2D(SourceTexture, LinearTextureSampler), inTexcoord, 0.0);
}
