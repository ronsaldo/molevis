#ifndef SPHERE_HPP
#define SPHERE_HPP

#include "Vector2.hpp"
#include "Vector3.hpp"
#include "Ray.hpp"

struct Sphere
{
    float radius;
    Vector3 center;

    bool rayIntersectionTest(const Ray &ray, Vector2 &lambdas)
    {
        // Ray sphere intersection formula from: https://viclw17.github.io/2018/07/16/raytracing-ray-sphere-intersection
        Vector3 rayOriginSphereCenter = ray.origin - center;
        float a = ray.direction.dot(ray.direction);
        float b = 2.0f * ray.direction.dot(rayOriginSphereCenter);
        float c = rayOriginSphereCenter.dot(rayOriginSphereCenter) - radius*radius;
        float delta = b*b - 4.0f*a*c;
        if (delta < 0.0f)
            return false;

        float deltaSqrt = sqrt(delta);
        lambdas = Vector2(-b - deltaSqrt, -b + deltaSqrt) / float(2.0*a);
        return true;
    }
};

#endif //SPHERE_HPP