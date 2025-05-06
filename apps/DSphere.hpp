#ifndef DSPHERE_HPP
#define DSPHERE_HPP

#include "DVector2.hpp"
#include "DVector3.hpp"
#include "DRay.hpp"

struct DSphere
{
    double radius;
    DVector3 center;

    bool rayIntersectionTest(const DRay &ray, DVector2 &lambdas) const
    {
        // Ray sphere intersection formula from: https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection
        DVector3 rayOriginSphereCenter = ray.origin - center;
        float a = ray.direction.dot(ray.direction);
        float b = 2.0f * ray.direction.dot(rayOriginSphereCenter);
        float c = rayOriginSphereCenter.dot(rayOriginSphereCenter) - radius*radius;
        float delta = b*b - 4.0f*a*c;
        if (delta < 0.0f)
            return false;

        float deltaSqrt = sqrt(delta);
        lambdas = DVector2(-b - deltaSqrt, -b + deltaSqrt) / float(2.0*a);
        return true;
    }
};

#endif //SPHERE_HPP