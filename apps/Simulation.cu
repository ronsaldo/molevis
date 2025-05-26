#include "Simulation.hpp"
#include <assert.h>

__global__
void resetNetForces(int atomCount, AtomSimulationState *atomStates)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        atomStates[i].netForce.x = 0;
        atomStates[i].netForce.y = 0;
        atomStates[i].netForce.z = 0;
    }
}

__global__
void integrateNetForces(int atomCount, AtomSimulationState *atomStates, double timeStep)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        double3 netForce = make_double3(atomStates[i].netForce.x, atomStates[i].netForce.y, atomStates[i].netForce.z);
        double3 startVelocity = make_double3(atomStates[i].velocity.x, atomStates[i].velocity.y, atomStates[i].velocity.z);
        
        double3 velocity = make_double3(
            startVelocity.x + netForce.x*timeStep,
            startVelocity.y + netForce.y*timeStep,
            startVelocity.z + netForce.z*timeStep
        );
        atomStates[i].velocity.x = velocity.x;
        atomStates[i].velocity.y = velocity.y;
        atomStates[i].velocity.z = velocity.z;

        double3 startPosition = make_double3(atomStates[i].position.x, atomStates[i].position.y, atomStates[i].position.z);
        double3 position = make_double3(
            startPosition.x + velocity.x*timeStep,
            startPosition.y + velocity.y*timeStep,
            startPosition.z + velocity.z*timeStep
        );

        atomStates[i].position.x = position.x;
        atomStates[i].position.y = position.y;
        atomStates[i].position.z = position.z;
    }
}

__global__
void computeLennardJonesForce(int atomCount, AtomDescription *atomDescriptions, AtomSimulationState *atomStates)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        AtomDescription &firstAtomDesc = atomDescriptions[i];
        AtomSimulationState &firstAtomState = atomStates[i];

        double3 netForce = make_double3(atomStates[i].netForce.x, atomStates[i].netForce.y, atomStates[i].netForce.z);
        double3 firstPosition = make_double3(firstAtomState.position.x, firstAtomState.position.y, firstAtomState.position.z);

        double firstLennardJonesCutoff  = firstAtomDesc.lennardJonesCutoff;
        double firstLennardJonesEpsilon = firstAtomDesc.lennardJonesEpsilon;
        double firstLennardJonesSigma   = firstAtomDesc.lennardJonesSigma;

        for(int j = 0; j < atomCount; ++j)
        {
            if(i == j)
                continue;

            AtomDescription &secondAtomDesc = atomDescriptions[j];
            AtomSimulationState &secondAtomState = atomStates[j];

            double3 secondPosition = make_double3(secondAtomState.position.x, secondAtomState.position.y, secondAtomState.position.z);

            double secondLennardJonesCutoff  = secondAtomDesc.lennardJonesCutoff;
            double secondLennardJonesEpsilon = secondAtomDesc.lennardJonesEpsilon;
            double secondLennardJonesSigma   = secondAtomDesc.lennardJonesSigma;

            double lennardJonesCutoff = firstLennardJonesCutoff + secondLennardJonesCutoff;
            double lennardJonesEpsilon = sqrt(firstLennardJonesEpsilon*secondLennardJonesEpsilon);
            double lennardJonesSigma = (firstLennardJonesSigma + secondLennardJonesSigma) * 0.5;
    
            double3 direction = make_double3(
                firstPosition.x - secondPosition.x,
                firstPosition.y - secondPosition.y,
                firstPosition.z - secondPosition.z
            );
            double dist2 = direction.x*direction.x + direction.y*direction.y + direction.z*direction.z;
            double dist = sqrt(dist2);
            if(1e-6 < dist && dist < lennardJonesCutoff)
            {
                double3 normalizedDirection = make_double3(
                    direction.x / dist,
                    direction.y / dist,
                    direction.z / dist
                );

                double derivative = lennardJonesDerivative(dist, lennardJonesSigma, lennardJonesEpsilon);
                double3 force = make_double3(
                    -normalizedDirection.x * derivative,
                    -normalizedDirection.y * derivative,
                    -normalizedDirection.z * derivative
                );

                netForce.x += force.x;
                netForce.y += force.y;
                netForce.z += force.z;
            }
        }
        
        atomStates[i].netForce.x = netForce.x;
        atomStates[i].netForce.y = netForce.y;
        atomStates[i].netForce.z = netForce.z;
    }
}

void performCudaSimulationStep(
    int atomDescriptionCount, AtomDescription *deviceAtomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationState *atomStates
)
{
    assert(atomDescriptionCount == atomStateSize);

    int blockSize = 256;    
    int blockCount = (atomStateSize + blockSize - 1) / blockSize; 

    // Reset the net forces
    resetNetForces<<<blockCount, blockSize>>> (atomStateSize, atomStates);

    // Reset the net forces
    computeLennardJonesForce<<<blockCount, blockSize>>> (atomStateSize, deviceAtomDescriptions, atomStates);

    // Integrate forces
    integrateNetForces<<<blockCount, blockSize>>> (atomStateSize, atomStates, SimulationTimeStep);
}
