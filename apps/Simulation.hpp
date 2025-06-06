#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "Potentials.hpp"
#include "AtomDescription.hpp"
#include "AtomBondDescription.hpp"
#include "AtomState.hpp"

#ifdef USE_CUDA
#include "cuda_runtime_api.h"

void performCudaSingleSimulationStep(
    int atomDescriptionCount, AtomDescription *deviceAtomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationSingleState *atomStates,
    float *kineticEnergyFrontBuffer, float *kineticEnergyBackBuffer,
    float targetTemperature
);

void performCudaDoubleSimulationStep(
    int atomDescriptionCount, AtomDescription *deviceAtomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationDoubleState *atomStates,
    double *kineticEnergyFrontBuffer, double *kineticEnergyBackBuffer,
    double targetTemperature
);
#endif // USE_CUDA
#endif //SIMULATION_HPP