#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include "Potentials.hpp"
#include "AtomDescription.hpp"
#include "AtomBondDescription.hpp"
#include "AtomState.hpp"
#include "cuda_runtime_api.h"

void performCudaSimulationStep(
    size_t atomDescriptionCount, AtomDescription *deviceAtomDescriptions,
    size_t atomBondDescriptionCount, AtomBondDescription *atomBondDescriptions,
    size_t atomStateSize, AtomSimulationState *atomStates
);

#endif //SIMULATION_HPP