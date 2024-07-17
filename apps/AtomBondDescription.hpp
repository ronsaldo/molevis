#ifndef MOLLEVIS_ATOM_BOND_DESCRIPTION_HPP
#define MOLLEVIS_ATOM_BOND_DESCRIPTION_HPP

#include "Vector4.hpp"
#include <stdint.h>

struct AtomBondDescription
{
    uint32_t firstAtomIndex;
    uint32_t secondAtomIndex;
    float morseEquilibriumDistance;
    float morseWellDepth;
    float thickness;
    Vector4 color;
};

#endif //MOLLEVIS_ATOM_BOND_DESCRIPTION_HPP