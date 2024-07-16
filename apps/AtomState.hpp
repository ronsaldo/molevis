#ifndef MOLLEVIS_ATOM_STATE_HPP
#define MOLLEVIS_ATOM_STATE_HPP

#include "Vector3.hpp"

struct AtomState
{
    Vector3 position;
    Vector3 velocity;
    Vector3 netForce;
};

#endif //MOLLEVIS_ATOM_STATE_HPP
