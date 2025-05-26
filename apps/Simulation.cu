#include "Simulation.hpp"

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

void performCudaSimulationStep(
    int atomDescriptionCount, AtomDescription *deviceAtomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationState *atomStates
)
{
    int blockSize = 256;    
    int blockCount = (atomStateSize + blockSize - 1) / blockSize; 

    // Reset the net forces
    resetNetForces<<<blockCount, blockSize>>> (atomStateSize, atomStates);

    // Integrate forces
    integrateNetForces<<<blockCount, blockSize>>> (atomStateSize, atomStates, SimulationTimeStep);
}
