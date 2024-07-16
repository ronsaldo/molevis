#version 450

layout(std140, set = 1, binding = 0) uniform CameraStateBlock
{
    uvec2 screenSize;

    bool flipVertically;
    float screenScale;

    mat4 projectionMatrix;
    mat4 viewMatrix;
    mat4 inverseViewMatrix;
} CameraState;

struct AtomDescription
{
    float radius;
    float mass;
    vec2 lennardJonesCoefficients;
    vec4 color;
};

struct AtomState
{
    vec3 position;
    vec3 velocity;
    vec3 netForce;
};

layout(std430, set = 2, binding = 0) buffer AtomDescriptionBufferBlock
{
    AtomDescription AtomDescriptionBuffer[];
};

layout(std430, set = 2, binding = 2) buffer AtomStateBufferBlock
{
    AtomState AtomStateBuffer[];
};

layout(location=0) flat in uint inAtomIndex;
layout(location=1) in vec3 inViewPosition;

layout(location=0) out vec4 outFragColor;

bool raySphereTest(float sphereRadius, in vec3 sphereCenter, in vec3 rayDirection, out vec2 lambdas)
{
    // Ray sphere intersection formula from: https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection/
    float a = dot(rayDirection, rayDirection);
    float b = 2.0 * dot(rayDirection, -sphereCenter);
    float c = dot(sphereCenter, sphereCenter) - sphereRadius*sphereRadius;
    float delta = b*b - 4.0*a*c;
    if (delta < 0.0)
        return false;

	float deltaSqrt = sqrt(delta);
    lambdas = vec2(-b - deltaSqrt, -b + deltaSqrt) / (2.0*a);

    return true;
}

void main()
{
    AtomDescription desc = AtomDescriptionBuffer[inAtomIndex];
    vec3 worldCenter = AtomStateBuffer[inAtomIndex].position;
    vec3 viewCenter = (CameraState.viewMatrix*vec4(worldCenter, 1.0)).xyz;

    vec3 D = normalize(inViewPosition);
    vec2 lambdas;
    //raySphereTest(desc.radius, viewCenter, D, lambdas);
    if(!raySphereTest(desc.radius, viewCenter, D, lambdas))
        discard;

    vec3 P = D*min(lambdas.x, lambdas.y);
    vec3 N = normalize(P - viewCenter);
    vec3 V = -D;

    // Premultiplied alpha
    vec4 baseColor = vec4(desc.color.rgb * desc.color.a, desc.color.a);
    float NdotV = max(0.0, dot(N, V));

    outFragColor = vec4(baseColor.rgb*(0.2 + NdotV*0.8), baseColor.a);

    //outFragColor = vec4(N.xyz*0.5 + 0.5, 1.0);
}
