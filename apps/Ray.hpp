#ifndef RAY_HPP
#define RAY_HPP

#include "Vector3.hpp"

struct Ray
{
    static Ray fromTo(const Vector3 &start, const Vector3 &end)
    {
        Ray ray;
        ray.origin = start;
        ray.direction = (end - start).normalized();
        ray.inverseDirection = ray.direction.reciprocal();
        return ray;
    }

    Vector3 origin;
    Vector3 direction;
    Vector3 inverseDirection;
};

#endif //SPHERE_HPP