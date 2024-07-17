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

struct ScreenBoundingQuad
{
    vec3 quadMin;
    bool inFrustum;
    vec3 quadMax;
};

layout(std430, set = 3, binding = 0) buffer ScreenBoundingQuadBufferBlock
{
    ScreenBoundingQuad ScreenBoundingQuadBuffer[];
};

layout(location = 0) flat out uint outAtomIndex;
layout(location = 1) out vec3 outViewPosition;

const vec2 quadVertices[4] = vec2[4](
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0)
);

void main()
{
    // Pass the instance index
    outAtomIndex = gl_InstanceIndex;

    // Fetch the screen quad.
    ScreenBoundingQuad boundingQuad = ScreenBoundingQuadBuffer[gl_InstanceIndex];
    if (!boundingQuad.inFrustum)
    {
        // Clipping by placing outsize the unit box
        outViewPosition = vec3(0.0);
        gl_Position = vec4(10.0, 10.0, 10.0, 1.0);
        return;
    }

    vec3 viewBoxMin = boundingQuad.quadMin;
    vec3 viewBoxMax = boundingQuad.quadMax;
    vec3 viewBoxExtent = viewBoxMax - viewBoxMin;

    // Compute the quad view position,
    vec2 quadCoord = quadVertices[gl_VertexIndex];
    vec3 viewPosition = vec3(viewBoxMin.xy + viewBoxExtent.xy*quadCoord, viewBoxMin.z);
    outViewPosition = viewPosition;

    // Perspective projection.
    gl_Position = CameraState.projectionMatrix * vec4(viewPosition, 1.0);
}