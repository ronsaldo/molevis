#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "Potentials.hpp"
#include "AtomDescription.hpp"
#include "AtomBondDescription.hpp"
#include "AtomState.hpp"
#include "cuda_runtime_api.h"

void performCudaSingleSimulationStep(
    int atomDescriptionCount, AtomDescription *deviceAtomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationSingleState *atomStates,
    float *kineticEnergyFrontBuffer, float *kineticEnergyBackBuffer
);

void performCudaDoubleSimulationStep(
    int atomDescriptionCount, AtomDescription *deviceAtomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationDoubleState *atomStates,
    double *kineticEnergyFrontBuffer, double *kineticEnergyBackBuffer
);

#endif //SIMULATION_HPP