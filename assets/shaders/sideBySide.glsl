#version 450
layout(set=0, binding=0) uniform sampler LinearTextureSampler;
layout(set=1, binding=4) uniform texture2D LeftEyeTexture;
layout(set=1, binding=5) uniform texture2D RightEyeTexture;

layout(location = 0) in vec2 inTexcoord;
layout(location = 0) out vec4 outColor;

void main()
{
	if(inTexcoord.x <= 0.5)
		outColor = textureLod(sampler2D(LeftEyeTexture, LinearTextureSampler), inTexcoord*vec2(2.0, 1.0), 0.0);
	else
		outColor = textureLod(sampler2D(RightEyeTexture, LinearTextureSampler), (inTexcoord - vec2(0.5, 0.0))*vec2(2.0, 1.0), 0.0);
}
