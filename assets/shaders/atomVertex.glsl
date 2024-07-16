#version 450

layout(std140, set = 1, binding = 0) uniform CameraStateBlock
{
    uvec2 screenSize;

    bool flipVertically;
    float screenScale;

    mat4 projectionMatrix;
    mat4 viewMatrix;
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

layout(location = 0) out vec4 outColor;
layout(location = 1) out vec3 outViewPosition;

const vec2 quadVertices[4] = vec2[4](
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0)
);

void main()
{
    // Fetch the instance data.
    AtomDescription desc = AtomDescriptionBuffer[gl_InstanceIndex];
    AtomState state = AtomStateBuffer[gl_InstanceIndex];

    // Compute the world bounding box
    vec3 worldHalfExtent = vec3(desc.radius);
    vec3 worldBoxCenter = state.position;
    vec3 worldBoxMin = worldBoxCenter - worldHalfExtent;
    vec3 worldBoxMax = worldBoxCenter + worldHalfExtent;

    // Compute the view bounding box
    vec3 viewBoxNeg = (CameraState.viewMatrix * vec4(worldBoxMin, 1.0)).xyz;
    vec3 viewBoxPos = (CameraState.viewMatrix * vec4(worldBoxMax, 1.0)).xyz;
    vec3 viewBoxMin = min(viewBoxNeg, viewBoxPos);
    vec3 viewBoxMax = max(viewBoxNeg, viewBoxPos);

    // Compute the extent of the box at the front.
    vec3 viewBoxFrontMin = vec3(viewBoxMin.xy, viewBoxMax.z);
    vec3 viewBoxFrontMax = vec3(viewBoxMax.xy, viewBoxMax.z);
    vec3 viewBoxFrontExtent = viewBoxFrontMax - viewBoxFrontMin;

    // Pass the instance color
    outColor = desc.color;

    // Compute the quad view position,
    vec2 quadCoord = quadVertices[gl_VertexIndex];
    vec3 viewPosition = vec3(viewBoxFrontMin.xy + viewBoxFrontExtent.xy*quadCoord, viewBoxFrontMin.z);
    outViewPosition = viewPosition;

    gl_Position = CameraState.projectionMatrix * vec4(viewPosition, 1.0);
}