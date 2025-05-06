#ifndef DDRay_HPP
#define DDRay_HPP

#include "DVector3.hpp"

struct DRay
{
    static DRay fromTo(const DVector3 &start, const DVector3 &end)
    {
        DRay DRay;
        DRay.origin = start;
        DRay.direction = (end - start).normalized();
        DRay.inverseDirection = DRay.direction.reciprocal();
        return DRay;
    }

    static DRay withOriginAndDirection(const DVector3 &origin, const DVector3 &direction)
    {
        DRay DRay;
        DRay.origin = origin;
        DRay.direction = direction;
        DRay.inverseDirection = direction.reciprocal();
        return DRay;
    }

    DVector3 origin;
    DVector3 direction;
    DVector3 inverseDirection;
};

#endif //SPHERE_HPP