#line 2

layout(location=0) flat in uint inAtomIndex;
layout(location=1) in vec3 inViewPosition;

layout(location=0) out vec4 outFragColor;

layout (depth_less) out float gl_FragDepth; // To keep early-z test.

bool raySphereTest(float sphereRadius, in vec3 sphereCenter, in vec3 rayDirection, out vec2 lambdas)
{
    // Ray sphere intersection formula from: https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection
    vec3 rayOriginSphereCenter = vec3(0.0) - sphereCenter;
    float a = dot(rayDirection, rayDirection);
    float b = 2.0 * dot(rayDirection, rayOriginSphereCenter);
    float c = dot(rayOriginSphereCenter, rayOriginSphereCenter) - sphereRadius*sphereRadius;
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
    vec3 worldCenter = (CameraState.atomModelMatrix*vec4(AtomStateBuffer[inAtomIndex].position, 1.0)).xyz;
    vec3 worldRadiusVertex = (CameraState.atomModelMatrix*vec4(AtomStateBuffer[inAtomIndex].position + vec3(desc.radius, 0.0, 0.0), 1.0)).xyz;
    float radius = length(worldRadiusVertex - worldCenter);
    vec3 viewCenter = (CameraState.viewMatrix*vec4(worldCenter, 1.0)).xyz;

    vec3 D = normalize(inViewPosition);
    vec2 lambdas;
    //raySphereTest(desc.radius, viewCenter, D, lambdas);
    bool inside = raySphereTest(radius, viewCenter, D, lambdas);
    if (!inside)
        discard;

    // Compute the intersection point.
    vec3 P = D*min(lambdas.x, lambdas.y);

    // Compute the intersection point depth.
    vec4 clipPosition = CameraState.projectionMatrix * vec4(P, 1.0);
    gl_FragDepth = clipPosition.z / clipPosition.w;

    // Compute the normal at the intersection point.
    vec3 N = normalize(P - viewCenter);
    vec3 V = -D;

    // Premultiplied alpha
    vec4 baseColor = vec4(desc.color.rgb * desc.color.a, desc.color.a);
    float NdotV = max(0.0, dot(N, V));

    outFragColor = vec4(baseColor.rgb*(0.2 + NdotV*0.8), baseColor.a);
    //if (!inside)
    //    outFragColor = vec4(1.0, 0.0, 1.0, 1.0);

    //outFragColor = vec4(N.xyz*0.5 + 0.5, 1.0);
    //outFragColor = vec4(V.xyz*0.5 + 0.5, 1.0);

    //outFragColor = gl_FragCoord.z - gl_FragDepth < 0.0 ? vec4(1.0, 1.0, 0.0, 1.0) : vec4(0.0, 1.0, 1.0, 1.0);
}
