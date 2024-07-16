#ifndef MOLLEVIS_VECTOR4_HPP
#define MOLLEVIS_VECTOR4_HPP

#include <math.h>
#include "Vector3.hpp"

struct alignas(16) Vector4
{
    float x, y, z, w;

    float dot(const Vector4 &o) const
    {
        return x*o.x + y*o.y + z*o.z + w*o.w;
    }

    Vector4 operator+(const Vector4 &o) const
    {
        return Vector4{x + o.x, y + o.y, z + o.z, w + o.w};
    }

    Vector4 operator-(const Vector4 &o) const
    {
        return Vector4{x - o.x, y - o.y, z - o.z, w - o.w};
    }

    Vector4 operator*(const Vector4 &o) const
    {
        return Vector4{x * o.x, y * o.y, z * o.z, w * o.w};
    }

    Vector4 operator/(const Vector4 &o) const
    {
        return Vector4{x / o.x, y / o.y, z / o.z, w / o.w};
    }

};

inline Vector4 Vector3::asVector4() const
{
    return Vector4{x, y, z, 0.0f};
}

#endif //MOLLEVIS_VECTOR4_HPP