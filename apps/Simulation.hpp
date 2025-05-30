#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "Potentials.hpp"
#include "AtomDescription.hpp"
#include "AtomBondDescription.hpp"
#include "AtomState.hpp"
#include "cuda_runtime_api.h"

void performCudaSimulationStep(
    int atomDescriptionCount, AtomDescription *deviceAtomDescriptions,
    int atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    int atomStateSize, AtomSimulationState *atomStates,
    double *kineticEnergyFrontBuffer, double *kineticEnergyBackBuffer
);

#endif //SIMULATION_HPP