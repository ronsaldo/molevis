#ifndef MOLLEVIS_DVECTOR3_HPP
#define MOLLEVIS_DVECTOR3_HPP

struct Vector4;

struct alignas(32) DVector3
{
    double x, y, z;

    DVector3() = default;
    DVector3(double cx, double cy, double cz)
        : x(cx), y(cy), z(cz) {}
    DVector3(double s)
        : x(s), y(s), z(s) {}

    DVector3 cross(const DVector3 &o) const
    {
        return DVector3{
            y*o.z - z*o.y,
            z*o.x - x*o.z,
            x*o.y - y*o.x
        };
    }

    double dot(const DVector3 &o) const
    {
        return x*o.x + y*o.y + z*o.z;
    }

    double length() const
    {
        return sqrt(x*x + y*y + z*z);
    }

    DVector3 operator-() const
    {
        return DVector3{-x, -y, -z};
    }

    DVector3 operator+(const DVector3 &o) const
    {
        return DVector3{x + o.x, y + o.y, z + o.z};
    }

    DVector3 operator-(const DVector3 &o) const
    {
        return DVector3{x - o.x, y - o.y, z - o.z};
    }

    DVector3 operator*(const DVector3 &o) const
    {
        return DVector3{x * o.x, y * o.y, z * o.z};
    }

    DVector3 operator/(const DVector3 &o) const
    {
        return DVector3{x / o.x, y / o.y, z / o.z};
    }

    DVector3 operator+=(const DVector3 &o)
    {
        return *this = *this + o;
    }

    Vector4 asVector4() const;

};

struct PackedDVector3
{
    double x, y, z;
};

#endif //MOLLEVIS_VECTOR3_HPP