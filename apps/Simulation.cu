#include "Simulation.hpp"
#include <assert.h>

__global__
void resetDoubleNetForces(int atomCount, AtomSimulationDoubleState *atomStates)
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
void integrateDoubleNetForces(int atomCount, AtomDescription *atomDescriptions, AtomSimulationDoubleState *atomStates, double timeStep)
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
void integrateDoubleVelocities(int atomCount, AtomDescription *atomDescriptions, AtomSimulationDoubleState *atomStates, double timeStep)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
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
void computeDoubleLennardJonesForce(int atomCount, AtomDescription *atomDescriptions, AtomSimulationDoubleState *atomStates)
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
            double dist = max(lennardJonesSigma, sqrt(dist2));
            if(dist < lennardJonesCutoff)
            {
                double3 normalizedDirection = make_double3(
                    direction.x / dist,
                    direction.y / dist,
                    direction.z / dist
                );

                double derivative = lennardJonesDoubleDerivative(dist, lennardJonesSigma, lennardJonesEpsilon);
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
void computeDoubleBondForce(int bondCount, AtomBondDescription *atomBondDescriptions, AtomSimulationDoubleState *atomStates)
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

        double hookPotentialDer = hookPotentialDoubleDerivative(distance, bond.equilibriumDistance, 100.0);
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
void computeDoubleKineticEnergy(int atomCount, AtomDescription *atomDescriptions, AtomSimulationDoubleState *atomStates, double *kineticEnergies)
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
void kineticDoubleEnergySum(int atomCount, double *kineticEnergies)
{
    double sum = 0.0;
    for(int i = 0; i < atomCount; ++i)
        sum += kineticEnergies[i];
    kineticEnergies[0] = sum;
}

__global__
void scaleDoubleVelocities(int atomCount, AtomSimulationDoubleState *atomStates, double *kineticEnergies, double targetTemperature)
{
    double currentTemperature = averageKineticEnergyToTemperatureDouble(kineticEnergies[0]);
    double kineticEnergyLambda = sqrt(targetTemperature / max(1e-6, currentTemperature));

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

void performCudaDoubleSimulationStep(
    int atomDescriptionCount, AtomDescription *atomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationDoubleState *atomStates,
    double *kineticEnergyFrontBuffer, double *kineticEnergyBackBuffer,
    double targetTemperature
)
{
    assert(atomDescriptionCount == atomStateSize);

    int blockSize = 32;    
    int blockCount = (atomStateSize + blockSize - 1) / blockSize; 

    int bondBlockSize = 32;
    int bondBlockCount = (atomBondDescriptionCount + bondBlockSize - 1) / bondBlockSize; 

    // Reset the net forces
    resetDoubleNetForces<<<blockCount, blockSize>>> (atomStateSize, atomStates);

    // Compute the lennard jones potential force.
    computeDoubleLennardJonesForce<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates);

    // Compute the bond forces.
    computeDoubleBondForce<<<bondBlockCount, bondBlockSize>>> (atomBondDescriptionCount, atomBondDescriptions, atomStates);

    // Integrate forces
    integrateDoubleNetForces<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates, SimulationTimeStep);

    // Compute kinetic energy
    computeDoubleKineticEnergy<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates, kineticEnergyFrontBuffer);

    // TODO: Use a parallel reduction.
    kineticDoubleEnergySum<<<atomStateSize, 1>>> (atomStateSize, kineticEnergyFrontBuffer);

    // Scale the velocities
    scaleDoubleVelocities<<<blockCount, blockSize>>> (atomStateSize, atomStates, kineticEnergyFrontBuffer, targetTemperature);

    // Integrate velocities
    integrateDoubleVelocities<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates, SimulationTimeStep);
}

__global__
void resetFloatNetForces(int atomCount, AtomSimulationSingleState *atomStates)
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
void integrateFloatNetForces(int atomCount, AtomDescription *atomDescriptions, AtomSimulationSingleState *atomStates, double timeStep)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        float mass = atomDescriptions[i].mass;
        float3 netForce = make_float3(atomStates[i].netForce.x, atomStates[i].netForce.y, atomStates[i].netForce.z);
        float3 startVelocity = make_float3(atomStates[i].velocity.x, atomStates[i].velocity.y, atomStates[i].velocity.z);
        
        float3 acceleration = make_float3(netForce.x/mass, netForce.y/mass, netForce.z/mass);
        float3 velocity = make_float3(
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
void integrateFloatVelocities(int atomCount, AtomDescription *atomDescriptions, AtomSimulationSingleState *atomStates, double timeStep)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        float3 velocity = make_float3(atomStates[i].velocity.x, atomStates[i].velocity.y, atomStates[i].velocity.z);

        float3 startPosition = make_float3(atomStates[i].position.x, atomStates[i].position.y, atomStates[i].position.z);
        float3 position = make_float3(
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
void computeFloatLennardJonesForce(int atomCount, AtomDescription *atomDescriptions, AtomSimulationSingleState *atomStates)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        AtomDescription &firstAtomDesc = atomDescriptions[i];
        AtomSimulationSingleState &firstAtomRenderingState = atomStates[i];

        float3 netForce = make_float3(atomStates[i].netForce.x, atomStates[i].netForce.y, atomStates[i].netForce.z);
        float3 firstPosition = make_float3(firstAtomRenderingState.position.x, firstAtomRenderingState.position.y, firstAtomRenderingState.position.z);

        float firstLennardJonesCutoff  = firstAtomDesc.lennardJonesCutoff;
        float firstLennardJonesEpsilon = firstAtomDesc.lennardJonesEpsilon;
        float firstLennardJonesSigma   = firstAtomDesc.lennardJonesSigma;

        for(int j = 0; j < atomCount; ++j)
        {
            if(i == j)
                continue;

            AtomDescription &secondAtomDesc = atomDescriptions[j];
            AtomSimulationSingleState &secondAtomRenderingState = atomStates[j];

            float3 secondPosition = make_float3(secondAtomRenderingState.position.x, secondAtomRenderingState.position.y, secondAtomRenderingState.position.z);

            float secondLennardJonesCutoff  = secondAtomDesc.lennardJonesCutoff;
            float secondLennardJonesEpsilon = secondAtomDesc.lennardJonesEpsilon;
            float secondLennardJonesSigma   = secondAtomDesc.lennardJonesSigma;

            float lennardJonesCutoff = firstLennardJonesCutoff + secondLennardJonesCutoff;
            float lennardJonesEpsilon = sqrt(firstLennardJonesEpsilon*secondLennardJonesEpsilon);
            float lennardJonesSigma = (firstLennardJonesSigma + secondLennardJonesSigma) * 0.5;
    
            float3 direction = make_float3(
                firstPosition.x - secondPosition.x,
                firstPosition.y - secondPosition.y,
                firstPosition.z - secondPosition.z
            );
            float dist2 = direction.x*direction.x + direction.y*direction.y + direction.z*direction.z;
            float dist = max(lennardJonesSigma, sqrt(dist2));
            if(dist < lennardJonesCutoff)
            {
                float3 normalizedDirection = make_float3(
                    direction.x / dist,
                    direction.y / dist,
                    direction.z / dist
                );

                float derivative = lennardJonesSingleDerivative(dist, lennardJonesSigma, lennardJonesEpsilon);
                float3 force = make_float3(
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
void computeFloatBondForce(int bondCount, AtomBondDescription *atomBondDescriptions, AtomSimulationSingleState *atomStates)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < bondCount; i += stride)
    {
        AtomBondDescription &bond = atomBondDescriptions[i];
        AtomSimulationSingleState &firstAtomRenderingState = atomStates[bond.firstAtomIndex];
        AtomSimulationSingleState &secondAtomRenderingState = atomStates[bond.secondAtomIndex];

        float3 direction = make_float3(
            firstAtomRenderingState.position.x - secondAtomRenderingState.position.x,
            firstAtomRenderingState.position.y - secondAtomRenderingState.position.y,
            firstAtomRenderingState.position.z - secondAtomRenderingState.position.z
        );
        float distance = sqrt(direction.x*direction.x + direction.y*direction.y + direction.z*direction.z);
        float3 normalizedDirection = make_float3(
            direction.x / distance,
            direction.y / distance,
            direction.z / distance
        );

        float hookPotentialDer = hookPotentialSingleDerivative(distance, bond.equilibriumDistance, 100.0);
        float3 force = make_float3(
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
void computeFloatKineticEnergy(int atomCount, AtomDescription *atomDescriptions, AtomSimulationSingleState *atomStates, float *kineticEnergies)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        AtomDescription &atomDesc = atomDescriptions[i];
        AtomSimulationSingleState &atomState = atomStates[i];

        float kineticEnergy = 0.5*atomDesc.mass * (
            atomState.velocity.x*atomState.velocity.x +
            atomState.velocity.y*atomState.velocity.y +
            atomState.velocity.z*atomState.velocity.z
        );
        kineticEnergies[i] = kineticEnergy / atomCount;
    }
}

__global__
void kineticFloatEnergySum(int atomCount, float *kineticEnergies)
{
    float sum = 0.0;
    for(int i = 0; i < atomCount; ++i)
        sum += kineticEnergies[i];
    kineticEnergies[0] = sum;
}

__global__
void scaleFloatVelocities(int atomCount, AtomSimulationSingleState *atomStates, float *kineticEnergies, float targetTemperature)
{
    float currentTemperature = averageKineticEnergyToTemperatureFloat(kineticEnergies[0]);
    float kineticEnergyLambda = sqrt(targetTemperature / max(1e-6, currentTemperature));

    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for(int i = index; i < atomCount; i += stride)
    {
        AtomSimulationSingleState &atomState = atomStates[i];
        atomState.velocity.x *= kineticEnergyLambda;
        atomState.velocity.y *= kineticEnergyLambda;
        atomState.velocity.z *= kineticEnergyLambda;
    }
}

void performCudaSingleSimulationStep(
    int atomDescriptionCount, AtomDescription *atomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationSingleState *atomStates,
    float *kineticEnergyFrontBuffer, float *kineticEnergyBackBuffer,
    float targetTemperature
)
{
    assert(atomDescriptionCount == atomStateSize);

    int blockSize = 32;    
    int blockCount = (atomStateSize + blockSize - 1) / blockSize; 

    int bondBlockSize = 32;
    int bondBlockCount = (atomBondDescriptionCount + bondBlockSize - 1) / bondBlockSize; 

    // Reset the net forces
    resetFloatNetForces<<<blockCount, blockSize>>> (atomStateSize, atomStates);

    // Compute the lennard jones potential force.
    computeFloatLennardJonesForce<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates);

    // Compute the bond forces.
    computeFloatBondForce<<<bondBlockCount, bondBlockSize>>> (atomBondDescriptionCount, atomBondDescriptions, atomStates);

    // Integrate forces
    integrateFloatNetForces<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates, SimulationTimeStep);

    // Compute kinetic energy
    computeFloatKineticEnergy<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates, kineticEnergyFrontBuffer);

    // TODO: Use a parallel reduction.
    kineticFloatEnergySum<<<atomStateSize, 1>>> (atomStateSize, kineticEnergyFrontBuffer);

    // Scale the velocities
    scaleFloatVelocities<<<blockCount, blockSize>>> (atomStateSize, atomStates, kineticEnergyFrontBuffer, targetTemperature);

    // Integrate velocities
    integrateFloatVelocities<<<blockCount, blockSize>>> (atomStateSize, atomDescriptions, atomStates, SimulationTimeStep);
}
