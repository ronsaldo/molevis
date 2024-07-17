#version 450

struct AtomState
{
    vec3 position;
    vec3 velocity;
    vec3 netForce;
};

layout(local_size_x = 32) in;

layout(std430, set = 2, binding = 2) buffer OldAtomStateBufferBlock
{
    AtomState OldAtomStateBuffer[];
};

layout(std430, set = 2, binding = 3) buffer NewAtomStateBufferBlock
{
    AtomState NewAtomStateBuffer[];
};

void main()
{
    uint atomIndex = gl_GlobalInvocationID.x;

    // Fetch the old state.
    AtomState state = OldAtomStateBuffer[atomIndex];

    // Reset the net force.
    state.netForce = vec3(0.0);

    // Store the reseted new state.
    NewAtomStateBuffer[atomIndex] = state;
}
