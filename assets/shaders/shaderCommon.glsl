#version 450

layout(std140, set = 1, binding = 0) uniform CameraStateBlock
{
    uvec2 screenSize;

    bool flipVertically;
    float nearDistance;
    float farDistance;
    float padding;

    mat4 projectionMatrix;
    mat4 inverseProjectionMatrix;
    mat4 viewMatrix;
    mat4 inverseViewMatrix;

    mat4 atomModelMatrix;
    mat4 atomInverseModelMatrix;
} CameraState;

layout(std140, set = 4, binding = 0) uniform ModelStateBlock
{
    mat4 modelMatrix;
    mat4 inverseModelMatrix;
} ModelState;

struct AtomDescription
{
    int atomNumber;
    float radius;
    float mass;
    float lennardJonesCutoff;
    float lennardJonesEpsilon;
    float lennardJonesSigma;
    vec4 color;
};

struct AtomRenderingState
{
    vec3 position;
};

struct AtomBondDesc
{
    uint firstAtomIndex;
    uint secondAtomIndex;
    float morseEquilibriumDistance;
    float morseWellDepth;
    float morseWellWidth;
    float thickness;
    vec4 color;
};

struct ScreenBoundingQuad
{
    vec3 quadMin;
    bool inFrustum;
    vec3 quadMax;
};


struct UIElementQuad
{
    vec2 position;
    vec2 size;
    vec4 color;
    
    bool isGlyph;

    vec2 fontPosition;
    vec2 fontSize;
};

layout(std430, set = 1, binding = 1) buffer UIDataBufferBlock
{
    UIElementQuad UIDataBuffer[];
};

layout(std430, set = 2, binding = 0) buffer AtomDescriptionBufferBlock
{
    AtomDescription AtomDescriptionBuffer[];
};

layout(std430, set = 2, binding = 1) buffer AtomBondDescriptionBufferBlock
{
    AtomBondDesc AtomBondDescriptionBuffer[];
};

layout(std430, set = 2, binding = 2) buffer AtomRenderingStateBufferBlock
{
    AtomRenderingState AtomRenderingStateBuffer[];
};

layout(std430, set = 3, binding = 0) buffer ScreenBoundingQuadBufferBlock
{
    ScreenBoundingQuad ScreenBoundingQuadBuffer[];
};

layout(push_constant) uniform PushConstants
{
    int currentHiglightedAtom;
};

float lennardJonesDerivative(float r, float sigma, float epsilon)
{
    return 24*epsilon*(pow(sigma, 6)/pow(r, 7) - 2.0*pow(sigma, 12)/pow(r, 13));
}
