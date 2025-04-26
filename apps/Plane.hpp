#ifndef USGAR_PLANE_HPP
#define USGAR_PLANE_HPP

#include "Vector3.hpp"
#include "Vector4.hpp"
#include "Plane.hpp"

struct Plane
{
    Plane() = default;
    Plane(float nx, float ny, float nz, float d)
        : normal(nx, ny, nz), distance(d) {}

    static Plane makeWithPoints(Vector3 p1, Vector3 p2, Vector3 p3)
    {
        auto u = p2 - p1;
        auto v = p3 - p1;
        auto n = u.cross(v).normalized();
        auto distance = n.dot(p1);
        return Plane(n.x, n.y, n.z, distance);
    }
    
    Vector4 asVector4Plane() const
    {
        return Vector4(normal.x, normal.y, normal.z, -distance);
    }

    float signedDistanceToPoint(const Vector3 &v)
    {
        return normal.x*v.x + normal.y*v.y + normal.z*v.z - distance;
    }

	PackedVector3 normal;
    float distance;
};

struct DPlane
{
    DPlane() = default;
    DPlane(double cnx, double cny, double cnz, double cdn)
        : nx(cnx), ny(cny), nz(cnz), dn(cdn) {}


    double signedDistanceToPoint(const Vector3 &v)
    {
        return nx*v.x + ny*v.y + nz*v.z + dn;
    }

	double nx, ny, nz, dn;
};

#endif //USGAR_PLANE_HPP
