#ifndef MOLLEVIS_VECTOR3_HPP
#define MOLLEVIS_VECTOR3_HPP

struct Vector4;

struct alignas(16) Vector3
{
    float x, y, z;

    Vector3() = default;
    Vector3(float cx, float cy, float cz)
        : x(cx), y(cy), z(cz) {}
    Vector3(float s)
        : x(s), y(s), z(s) {}

    Vector3 cross(const Vector3 &o) const
    {
        return Vector3{
            y*o.z - z*o.y,
            z*o.x - x*o.z,
            x*o.y - y*o.x
        };
    }

    float dot(const Vector3 &o) const
    {
        return x*o.x + y*o.y + z*o.z;
    }

    Vector3 operator-() const
    {
        return Vector3{-x, -y, -z};
    }

    Vector3 operator+(const Vector3 &o) const
    {
        return Vector3{x + o.x, y + o.y, z + o.z};
    }

    Vector3 operator-(const Vector3 &o) const
    {
        return Vector3{x - o.x, y - o.y, z - o.z};
    }

    Vector3 operator*(const Vector3 &o) const
    {
        return Vector3{x * o.x, y * o.y, z * o.z};
    }

    Vector3 operator/(const Vector3 &o) const
    {
        return Vector3{x / o.x, y / o.y, z / o.z};
    }

    Vector4 asVector4() const;

};

#endif //MOLLEVIS_VECTOR3_HPP