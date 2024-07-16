#ifndef MOLLEVIS_VECTOR4_HPP
#define MOLLEVIS_VECTOR4_HPP

#include <math.h>
#include "Vector3.hpp"

struct alignas(16) Vector4
{
    float x, y, z, w;

    Vector4() = default;
    Vector4(float cx, float cy, float cz, float cw)
        : x(cx), y(cy), z(cz), w(cw) {}
    Vector4(float s)
        : x(s), y(s), z(s), w(s) {}

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

    Vector3 minorAt(int index) const
    {
        switch(index)
        {
        case 0: return Vector3{y, z, w};
        case 1: return Vector3{x, z, w};
        case 2: return Vector3{x, y, w};
        case 3: return Vector3{x, y, z};
        default: abort();
        }
    }
};

inline Vector4 Vector3::asVector4() const
{
    return Vector4{x, y, z, 0.0f};
}

#endif //MOLLEVIS_VECTOR4_HPP