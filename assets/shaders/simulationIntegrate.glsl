#version 450

struct AtomState
{
    vec3 position;
    vec3 velocity;
    vec3 netForce;
};

struct AtomDescription
{
    float radius;
    float mass;
    vec2 lennardJonesCoefficients;
    vec4 color;
};

layout(local_size_x = 32) in;

layout(std430, set = 2, binding = 0) buffer AtomDescriptionBufferBlock
{
    AtomDescription AtomDescriptionBuffer[];
};


layout(std430, set = 2, binding = 3) buffer AtomStateBufferBlock
{
    AtomState AtomStateBuffer[];
};

layout(push_constant) uniform PushConstants
{
    float timeStep;
    uint atomCount;
    uint bondCount;
};

vec3 floorMod(vec3 position, float scale)
{
    return position - floor(position / scale)*scale;
}

void main()
{
    uint atomIndex = gl_GlobalInvocationID.x;

    // Fetch the old state.
    AtomState state = AtomStateBuffer[atomIndex];
    float mass = AtomDescriptionBuffer[atomIndex].mass;

    // Symplectic
    vec3 acceleration = state.netForce / mass;
    state.velocity += acceleration*timeStep;
    state.position += state.velocity*timeStep;

    //state.position = floorMod(state.position, 100.0);

    // Write the integrated state
    AtomStateBuffer[atomIndex] = state;
}
