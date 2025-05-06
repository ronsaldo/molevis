#ifndef MOLLEVIS_ATOM_STATE_HPP
#define MOLLEVIS_ATOM_STATE_HPP

#include "Vector3.hpp"
#include "DVector3.hpp"

struct AtomState
{
    Vector3 position;
};

struct AtomSimulationState
{
    DVector3 position;
    DVector3 velocity;
    DVector3 netForce;

    AtomState asRenderingState()
    {
        return AtomState{
            {float(position.x), float(position.y), float(position.z)},
        };
    }
};

#endif //MOLLEVIS_ATOM_STATE_HPP
