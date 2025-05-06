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

    double length2() const
    {
        return x*x + y*y + z*z;
    }

    double length() const
    {
        return sqrt(x*x + y*y + z*z);
    }

    DVector3 max(const DVector3 &o) const
    {
        return DVector3(
            std::max(x, o.x),
            std::max(x, o.y),
            std::max(x, o.z)
        );
    }

    DVector3 abs() const
    {
        return max(-*this);
    }

    DVector3 normalized() const
    {
        auto l = length();
        if(l < 1e-6)
            return DVector3(0);
        return DVector3(x / l, y/ l, z / l);
    }

    DVector3 reciprocal() const
    {
        return DVector3(1.0f/x, 1.0f/y, 1.0f/z);
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