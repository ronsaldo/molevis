#line 2

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexcoord;

layout(location = 0) out vec4 outColor;

layout(set=0, binding=0) uniform sampler LinearTextureSampler;
layout(set=4, binding=1) uniform texture2D ModelTexture;

void main()
{
    vec3 N = normalize(inNormal);
    vec3 V = normalize(-inPosition);
    if(!gl_FrontFacing)
        N = -N;

    vec4 baseColor = texture(sampler2D(ModelTexture, LinearTextureSampler), inTexcoord);

    float NdotV = max(dot(N, V), 0.0);
    outColor = vec4(baseColor.rgb*NdotV, baseColor.a);
}
