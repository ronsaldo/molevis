#line 2

layout(location=0) flat in uint inBondIndex;
layout(location=1) in vec3 inViewPosition;

layout(location=0) out vec4 outFragColor;

// Capsule SDF from https://iquilezles.org/articles/distfunctions/ 
float sdfCapsule( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

vec3 sdfCapsuleGradient(vec3 p, vec3 a, vec3 b, float r)
{
    float dh = 0.0001;
    return normalize(vec3(
        sdfCapsule(p + vec3(dh, 0.0, 0.0), a, b, r) - sdfCapsule(p - vec3(dh, 0.0, 0.0), a, b, r),
        sdfCapsule(p + vec3(0.0, dh, 0.0), a, b, r) - sdfCapsule(p - vec3(0.0, dh, 0.0), a, b, r),
        sdfCapsule(p + vec3(0.0, 0.0, dh), a, b, r) - sdfCapsule(p - vec3(0.0, 0.0, dh), a, b, r)
    ));
}

void main()
{
    AtomBondDesc desc = AtomBondDescriptionBuffer[inBondIndex];
    vec3 firstAtomWorldPosition = (CameraState.atomModelMatrix * vec4(AtomStateBuffer[desc.firstAtomIndex].position, 1.0)).xyz;
    vec3 firstAtomWorldRadius = (CameraState.atomModelMatrix * (
        vec4(AtomStateBuffer[desc.firstAtomIndex].position, 1.0)
        + vec4(desc.thickness, 0.0, 0.0, 1.0))).xyz;

    vec3 secondAtomWorldPosition = (CameraState.atomModelMatrix *vec4(AtomStateBuffer[desc.secondAtomIndex].position, 1.0)).xyz;

    float radius = length(firstAtomWorldRadius - firstAtomWorldPosition);
    vec3 firstAtomViewPosition = (CameraState.viewMatrix * vec4(firstAtomWorldPosition, 1.0)).xyz;
    vec3 secondAtomViewPosition = (CameraState.viewMatrix * vec4(secondAtomWorldPosition, 1.0)).xyz;
    
    vec3 D = normalize(inViewPosition);
    vec3 viewPosition = D*CameraState.nearDistance;
    for(int i = 0; i < 4; ++i)
        viewPosition += sdfCapsule(viewPosition, firstAtomViewPosition, secondAtomViewPosition, radius)*D;

    vec3 V = -D;
    vec3 N = sdfCapsuleGradient(viewPosition, firstAtomViewPosition, secondAtomViewPosition, radius);

    // Premultiplied alpha
    vec4 baseColor = vec4(desc.color.rgb * desc.color.a, desc.color.a);
    float NdotV = max(0.0, dot(N, V));

    outFragColor = vec4(baseColor.rgb*(0.2 + NdotV*0.8), baseColor.a);
    //if (!inside)
    //    outFragColor = vec4(1.0, 0.0, 1.0, 1.0);

    //outFragColor = vec4(N.xyz*0.5 + 0.5, 1.0);
    //outFragColor = vec4(V.xyz*0.5 + 0.5, 1.0);
}
