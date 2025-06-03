#include "Simulation.hpp"
#include <assert.h>

__global__
void resetNetForces(int atomCount, AtomSimulationDoubleState *atomStates)
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
void integrateNetForces(int atomCount, AtomDescription *atomDescriptions, AtomSimulationDoubleState *atomStates, double timeStep)
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
void integrateVelocities(int atomCount, AtomDescription *atomDescriptions, AtomSimulationDoubleState *atomStates, double timeStep)
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
void computeLennardJonesForce(int atomCount, AtomDescription *atomDescriptions, AtomSimulationDoubleState *atomStates)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        AtomDescription &firstAtomDesc = atomDescriptions[i];
        AtomSimulationDoubleState &firstAtomRenderingState = atomStates[i];

        double3 netForce = make_double3(atomStates[i].netForce.x, atomStates[i].netForce.y, atomStates[i].netForce.z);
        double3 firstPosition = make_double3(firstAtomRenderingState.position.x, firstAtomRenderingState.position.y, firstAtomRenderingState.position.z);

        double firstLennardJonesCutoff  = firstAtomDesc.lennardJonesCutoff;
        double firstLennardJonesEpsilon = firstAtomDesc.lennardJonesEpsilon;
        double firstLennardJonesSigma   = firstAtomDesc.lennardJonesSigma;

        for(int j = 0; j < atomCount; ++j)
        {
            if(i == j)
                continue;

            AtomDescription &secondAtomDesc = atomDescriptions[j];
            AtomSimulationDoubleState &secondAtomRenderingState = atomStates[j];

            double3 secondPosition = make_double3(secondAtomRenderingState.position.x, secondAtomRenderingState.position.y, secondAtomRenderingState.position.z);

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
void computeBondForce(int bondCount, AtomBondDescription *atomBondDescriptions, AtomSimulationDoubleState *atomStates)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < bondCount; i += stride)
    {
        AtomBondDescription &bond = atomBondDescriptions[i];
        AtomSimulationDoubleState &firstAtomRenderingState = atomStates[bond.firstAtomIndex];
        AtomSimulationDoubleState &secondAtomRenderingState = atomStates[bond.secondAtomIndex];

        double3 direction = make_double3(
            firstAtomRenderingState.position.x - secondAtomRenderingState.position.x,
            firstAtomRenderingState.position.y - secondAtomRenderingState.position.y,
            firstAtomRenderingState.position.z - secondAtomRenderingState.position.z
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

        atomicAdd(&firstAtomRenderingState.netForce.x, force.x);
        atomicAdd(&firstAtomRenderingState.netForce.y, force.y);
        atomicAdd(&firstAtomRenderingState.netForce.z, force.z);

        atomicAdd(&secondAtomRenderingState.netForce.x, -force.x);
        atomicAdd(&secondAtomRenderingState.netForce.y, -force.y);
        atomicAdd(&secondAtomRenderingState.netForce.z, -force.z);
    }
}

__global__
void computeKineticEnergy(int atomCount, AtomDescription *atomDescriptions, AtomSimulationDoubleState *atomStates, double *kineticEnergies)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        AtomDescription &atomDesc = atomDescriptions[i];
        AtomSimulationDoubleState &atomState = atomStates[i];

        double kineticEnergy = 0.5*atomDesc.mass * (
            atomState.velocity.x*atomState.velocity.x +
            atomState.velocity.y*atomState.velocity.y +
            atomState.velocity.z*atomState.velocity.z
        );
        kineticEnergies[i] = kineticEnergy / atomCount;
    }
}

__global__
void kineticEnergySum(int atomCount, double *kineticEnergies)
{
    double sum = 0.0;
    for(int i = 0; i < atomCount; ++i)
        sum += kineticEnergies[i];
    kineticEnergies[0] = sum;
}

__global__
void scaleVelocities(int atomCount, AtomSimulationDoubleState *atomStates, double *kineticEnergies, double targetKineticEnergy)
{
    double kineticEnergyLambda = targetKineticEnergy / max(0.01, kineticEnergies[0]);

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        AtomSimulationDoubleState &atomState = atomStates[i];
        atomState.velocity.x *= kineticEnergyLambda;
        atomState.velocity.y *= kineticEnergyLambda;
        atomState.velocity.z *= kineticEnergyLambda;
    }
}

void performCudaSimulationStep(
    int atomDescriptionCount, AtomDescription *atomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationDoubleState *atomStates,
    double *kineticEnergyFrontBuffer, double *kineticEnergyBackBuffer
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

    // Compute kinetic energy
    computeKineticEnergy<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates, kineticEnergyFrontBuffer);

    // TODO: Use a parallel reduction.
    kineticEnergySum<<<atomStateSize, 1>>> (atomStateSize, kineticEnergyFrontBuffer);

    // Scale the velocities
    scaleVelocities<<<blockCount, blockSize>>> (atomStateSize, atomStates, kineticEnergyFrontBuffer, 1.0);

    // Integrate velocities
    integrateVelocities<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates, SimulationTimeStep);
}
