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

    // Get the atom world position
    vec3 worldCenter = state.position;
    
    // Compute the atom view position center.
    vec3 viewCenter = (CameraState.viewMatrix * vec4(worldCenter, 1.0)).xyz;

    // Compute the view bounding box
    vec3 viewBoxMin = viewCenter - desc.radius;
    vec3 viewBoxMax = viewCenter + desc.radius;

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