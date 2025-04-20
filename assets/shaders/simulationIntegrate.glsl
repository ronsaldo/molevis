#line 2

layout(local_size_x = 32) in;

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
