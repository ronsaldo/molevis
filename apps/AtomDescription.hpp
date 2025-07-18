#ifndef MOLLEVIS_ATOM_DESCRIPTION_HPP
#define MOLLEVIS_ATOM_DESCRIPTION_HPP

#include "Vector4.hpp"

struct AtomDescription
{
    int atomNumber;
    float radius;
    float mass;
    float lennardJonesCutoff;

    float lennardJonesEpsilon;
    float lennardJonesSigma;
    int isRigid;

    Vector4 color;
};

#endif //MOLLEVIS_ATOM_DESCRIPTION_HPP