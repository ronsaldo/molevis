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
void integrateNetForces(int atomCount, AtomDescription *atomDescriptions, AtomSimulationState *atomStates, double timeStep)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        double mass = atomDescriptions[i].mass;
        double3 netForce = make_double3(atomStates[i].netForce.x, atomStates[i].netForce.y, atomStates[i].netForce.z);
        double3 startVelocity = make_double3(atomStates[i].velocity.x, atomStates[i].velocity.y, atomStates[i].velocity.z);
        
        double3 acceleration = make_double3(netForce.x/mass, netForce.y/mass, netForce.z/mass);
        double3 velocity = make_double3(
            startVelocity.x + acceleration.x*timeStep,
            startVelocity.y + acceleration.y*timeStep,
            startVelocity.z + acceleration.z*timeStep
        );
        atomStates[i].velocity.x = velocity.x;
        atomStates[i].velocity.y = velocity.y;
        atomStates[i].velocity.z = velocity.z;
    }
}

__global__
void integrateVelocities(int atomCount, AtomDescription *atomDescriptions, AtomSimulationState *atomStates, double timeStep)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        double mass = atomDescriptions[i].mass;
        double3 velocity = make_double3(atomStates[i].velocity.x, atomStates[i].velocity.y, atomStates[i].velocity.z);

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

__global__
void computeBondForce(int bondCount, AtomBondDescription *atomBondDescriptions, AtomSimulationState *atomStates)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < bondCount; i += stride)
    {
        AtomBondDescription &bond = atomBondDescriptions[i];
        AtomSimulationState &firstAtomState = atomStates[bond.firstAtomIndex];
        AtomSimulationState &secondAtomState = atomStates[bond.secondAtomIndex];

        double3 direction = make_double3(
            firstAtomState.position.x - secondAtomState.position.x,
            firstAtomState.position.y - secondAtomState.position.y,
            firstAtomState.position.z - secondAtomState.position.z
        );
        double distance = sqrt(direction.x*direction.x + direction.y*direction.y + direction.z*direction.z);
        double3 normalizedDirection = make_double3(
            direction.x / distance,
            direction.y / distance,
            direction.z / distance
        );

        double hookPotentialDer = hookPotentialDerivative(distance, bond.equilibriumDistance, 100.0);
        double3 force = make_double3(
            -normalizedDirection.x*hookPotentialDer,
            -normalizedDirection.y*hookPotentialDer,
            -normalizedDirection.z*hookPotentialDer
        );

        atomicAdd(&firstAtomState.netForce.x, force.x);
        atomicAdd(&firstAtomState.netForce.y, force.y);
        atomicAdd(&firstAtomState.netForce.z, force.z);

        atomicAdd(&secondAtomState.netForce.x, -force.x);
        atomicAdd(&secondAtomState.netForce.y, -force.y);
        atomicAdd(&secondAtomState.netForce.z, -force.z);
    }
}

void performCudaSimulationStep(
    int atomDescriptionCount, AtomDescription *atomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationState *atomStates
)
{
    assert(atomDescriptionCount == atomStateSize);

    int blockSize = 256;    
    int blockCount = (atomStateSize + blockSize - 1) / blockSize; 

    int bondBlockSize = 256;    
    int bondBlockCount = (atomBondDescriptionCount + bondBlockSize - 1) / bondBlockSize; 

    // Reset the net forces
    resetNetForces<<<blockCount, blockSize>>> (atomStateSize, atomStates);

    // Compute the lennard jones potential force.
    computeLennardJonesForce<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates);

    // Compute the bond forces.
    computeBondForce<<<bondBlockCount, bondBlockSize>>> (atomBondDescriptionCount, atomBondDescriptions, atomStates);

    // Integrate forces
    integrateNetForces<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates, SimulationTimeStep);

    // Integrate velocities
    integrateVelocities<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates, SimulationTimeStep);
}
