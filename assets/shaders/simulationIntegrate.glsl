#version 450

struct AtomState
{
    vec3 position;
    vec3 velocity;
    vec3 netForce;
};

layout(local_size_x = 32) in;

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

void main()
{
    uint atomIndex = gl_GlobalInvocationID.x;

    // Fetch the old state.
    AtomState state = AtomStateBuffer[atomIndex];

    // Integrate the position and the velocity.
    state.position += state.velocity*timeStep + state.netForce*(timeStep*timeStep*0.5);
    state.velocity += state.netForce*timeStep;

    // Write the integrated state
    AtomStateBuffer[atomIndex] = state;
}
