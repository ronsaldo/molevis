#version 450

struct AtomDescription
{
    float radius;
    float mass;
    vec2 lennardJonesCoefficients;
    vec4 color;
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

struct AtomState
{
    vec3 position;
    vec3 velocity;
    vec3 netForce;
};

layout(local_size_x = 32) in;

layout(std430, set = 2, binding = 0) buffer AtomDescriptionBufferBlock
{
    AtomDescription AtomDescriptionBuffer[];
};

layout(std430, set = 2, binding = 1) buffer AtomBondDescriptionBufferBlock
{
    AtomBondDesc AtomBondDescriptionBuffer[];
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

float lennardJonesDerivative(float ir, float epsilon, float sigma)
{
    float sigma_ir = sigma * ir;
    float sigma_ir6 = pow(sigma_ir, 6.0);
    float sigma_ir12 = sigma_ir6*sigma_ir6;
    return 24.0*epsilon*(sigma_ir6 - 2.0*sigma_ir12) * ir;
}

float morsePotentialDerivative(float r, float D, float a, float re)
{
    float innerExp = exp(-a*(r - re));
    return -2.0*a*D*(1.0 - innerExp)*innerExp;
}

void main()
{
    uint myAtomIndex = gl_GlobalInvocationID.x;
    vec3 myAtomPosition = AtomStateBuffer[myAtomIndex].position;
    vec3 myNetForce = AtomStateBuffer[myAtomIndex].netForce;

    // Lennard-jones potential.
    for(uint firstAtomIndex = 0u; firstAtomIndex < atomCount; ++firstAtomIndex)
    {
        uint secondAtomIndex = myAtomIndex;
        vec3 secondPosition = myAtomPosition;

        if(firstAtomIndex == secondAtomIndex || firstAtomIndex >= atomCount || secondAtomIndex >= atomCount)
            continue;

        // Fetch the first position and the lennard jones coefficients.
        vec3 firstPosition = AtomStateBuffer[firstAtomIndex].position;
        vec2 coefficients = AtomDescriptionBuffer[firstAtomIndex].lennardJonesCoefficients;

        vec3 direction = secondPosition - firstPosition;
        float dist = length(direction);
        if(dist > 0.000001)
        {
            float invDist = 1.0 / dist;
            direction *= invDist;

            vec3 force = -direction * lennardJonesDerivative(invDist, coefficients.x, coefficients.y);
            myNetForce += force;
        }
    }

    // Bond morse potential.
    for(uint bondIndex = 0u; bondIndex < bondCount; ++bondIndex)
    {
        if(AtomBondDescriptionBuffer[bondIndex].secondAtomIndex != myAtomIndex)
            continue;

        uint secondAtomIndex = myAtomIndex;
        vec3 secondPosition = myAtomPosition;

        uint firstAtomIndex = AtomBondDescriptionBuffer[bondIndex].firstAtomIndex;
        vec3 firstPosition = AtomStateBuffer[firstAtomIndex].position;

        float morseEquilibriumDistance = AtomBondDescriptionBuffer[bondIndex].morseEquilibriumDistance;
        float morseWellDepth = AtomBondDescriptionBuffer[bondIndex].morseWellDepth;
        float morseWellWidth = AtomBondDescriptionBuffer[bondIndex].morseWellWidth;

        vec3 direction = secondPosition - firstPosition;
        float dist = length(direction);
        if(dist > 0.000001)
        {
            direction /= dist;

            vec3 force = -direction * morsePotentialDerivative(dist, morseWellDepth, morseWellWidth, morseEquilibriumDistance);
            myNetForce += force;
        }
    }

    AtomStateBuffer[myAtomIndex].netForce = myNetForce;
}
