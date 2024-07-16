#version 450

layout(std140, set = 1, binding = 0) uniform CameraStateBlock
{
    uvec2 screenSize;

    bool flipVertically;
    float nearDistance;
    float farDistance;

    mat4 projectionMatrix;
    mat4 inverseProjectionMatrix;
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

layout(location = 0) flat out uint outAtomIndex;
layout(location = 1) out vec3 outViewPosition;

const vec2 quadVertices[4] = vec2[4](
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0)
);

vec3 projectViewPoint(vec3 point)
{
    vec4 clipPoint = CameraState.projectionMatrix * vec4(point, 1.0);
    return clipPoint.xyz / clipPoint.w;
}

vec3 unprojectPoint(vec3 point)
{
    vec4 clipPoint = CameraState.inverseProjectionMatrix * vec4(point, 1.0);
    return clipPoint.xyz / clipPoint.w;
}

void computeBoundingQuadForViewBoundingBox(vec3 viewBoundMin, vec3 viewBoundMax, out vec3 boundMin, out vec3 boundMax)
{
    vec3 viewBoundCorners[8] = vec3[8](
        vec3(viewBoundMin.x, viewBoundMin.y, viewBoundMin.z),
        vec3(viewBoundMax.x, viewBoundMin.y, viewBoundMin.z),
        vec3(viewBoundMin.x, viewBoundMax.y, viewBoundMin.z),
        vec3(viewBoundMax.x, viewBoundMax.y, viewBoundMin.z),

        vec3(viewBoundMin.x, viewBoundMin.y, viewBoundMax.z),
        vec3(viewBoundMax.x, viewBoundMin.y, viewBoundMax.z),
        vec3(viewBoundMin.x, viewBoundMax.y, viewBoundMax.z),
        vec3(viewBoundMax.x, viewBoundMax.y, viewBoundMax.z)
    );

    vec3 projectedMin;
    vec3 projectedMax;
    projectedMin = projectedMax = projectViewPoint(viewBoundCorners[0]);
    for(int i = 1; i < 8; ++i)
    {
        vec3 projectedPoint = projectViewPoint(viewBoundCorners[i]);
        projectedMin = min(projectedMin, projectedPoint);
        projectedMax = max(projectedMax, projectedPoint);
    }

    float projectedCornerDepth = projectedMin.z;
    vec3 projectedCornerMin = vec3(projectedMin.xy, projectedCornerDepth);
    vec3 projectedCornerMax = vec3(projectedMax.xy, projectedCornerDepth);

    vec3 viewProjectedCornerMin = unprojectPoint(projectedCornerMin);
    vec3 viewProjectedCornerMax = unprojectPoint(projectedCornerMax);
    boundMin = min(viewProjectedCornerMin, viewProjectedCornerMax);
    boundMax = max(viewProjectedCornerMin, viewProjectedCornerMax);
}

void computeBoundingQuadForViewSphere(vec3 center, float radius, out vec3 boundMin, out vec3 boundMax)
{
    computeBoundingQuadForViewBoundingBox(center - radius, center + radius, boundMin, boundMax);
}

void main()
{
    // Fetch the instance data.
    AtomDescription desc = AtomDescriptionBuffer[gl_InstanceIndex];
    AtomState state = AtomStateBuffer[gl_InstanceIndex];

    // Get the atom world position
    vec3 worldCenter = state.position;
    
    // Compute the atom view position center.
    float radius = desc.radius;
    vec3 viewCenter = (CameraState.viewMatrix * vec4(worldCenter, 1.0)).xyz;

    // Compute the extent of the box at the front.
    vec3 viewBoxFrontMin;
    vec3 viewBoxFrontMax;
    computeBoundingQuadForViewSphere(viewCenter, radius, viewBoxFrontMin, viewBoxFrontMax);
    vec3 viewBoxFrontExtent = viewBoxFrontMax - viewBoxFrontMin;

    // Pass the instance index
    outAtomIndex = gl_InstanceIndex;

    // Compute the quad view position,
    vec2 quadCoord = quadVertices[gl_VertexIndex];
    vec3 viewPosition = vec3(viewBoxFrontMin.xy + viewBoxFrontExtent.xy*quadCoord, viewBoxFrontMin.z);
    outViewPosition = viewPosition;

    gl_Position = CameraState.projectionMatrix * vec4(viewPosition, 1.0);
}