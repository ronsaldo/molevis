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
}