#line 2

layout(local_size_x = 32) in;

void main()
{
    uint atomIndex = gl_GlobalInvocationID.x;

    // Fetch the old state.
    AtomState state = OldAtomStateBuffer[atomIndex];

    // Reset the net force.
    state.netForce = vec3(0.0);

    // Store the reseted new state.
    AtomStateBuffer[atomIndex] = state;
}
