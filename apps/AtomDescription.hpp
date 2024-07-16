#ifndef MOLLEVIS_ATOM_DESCRIPTION_HPP
#define MOLLEVIS_ATOM_DESCRIPTION_HPP

#include "Vector4.hpp"

struct AtomDescription
{
    float radius;
    float mass;
    float lennardJonesA;
    float lennardJonesB;
    Vector4 color;
};

#endif //MOLLEVIS_ATOM_DESCRIPTION_HPP