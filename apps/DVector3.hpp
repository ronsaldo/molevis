#ifndef MOLLEVIS_DVECTOR3_HPP
#define MOLLEVIS_DVECTOR3_HPP

#include "VectorCommon.hpp"

struct Vector4;

struct alignas(32) DVector3
{
    union
    {
        struct {
            double x, y, z;
        };
#ifdef USE_AVX_VECTORS
        __m256d avxVector;
#endif
    };

    DVector3() = default;
#ifdef USE_AVX_VECTORS
    DVector3(__m256d cavxVector)
        : avxVector(cavxVector) {}
    DVector3(double cx, double cy, double cz)
        : avxVector(_mm256_set_pd(0, cz, cy, cx)) {}
    DVector3(double s)
        : avxVector(_mm256_broadcast_sd(&s)) {}

#else
    DVector3(double cx, double cy, double cz)
        : x(cx), y(cy), z(cz) {}
    DVector3(double s)
        : x(s), y(s), z(s) {}
#endif

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
        return ((*this) * o).sum();
    }

    double sum() const
    {
        return x + y + z;
    }

    double length2() const
    {
        return dot(*this);
    }

    double length() const
    {
        return sqrt(length2());
    }

    DVector3 max(const DVector3 &o) const
    {
#ifdef USE_AVX_VECTORS
        return DVector3(_mm256_max_pd(this->avxVector, o.avxVector));
#else
        return DVector3(
            std::max(x, o.x),
            std::max(x, o.y),
            std::max(x, o.z)
        );
#endif 
    }

    DVector3 min(const DVector3 &o) const
    {
#ifdef USE_AVX_VECTORS
        return DVector3(_mm256_min_pd(this->avxVector, o.avxVector));
#else
        return DVector3(
            std::min(x, o.x),
            std::min(x, o.y),
            std::min(x, o.z)
        );
#endif 
    }

    DVector3 abs() const
    {
        return max(-*this);
    }

    DVector3 normalized() const
    {
        auto l = length();
        if(l == 0)
            return DVector3(0);
        return *this / DVector3(l);
    }

    DVector3 reciprocal() const
    {
        return DVector3(1.0) / *this;
    }

#ifdef USE_AVX_VECTORS
    DVector3 operator-() const
    {
        return DVector3(0.0) - *this;
    }

    DVector3 operator+(const DVector3 &o) const
    {
        return DVector3(_mm256_add_pd(avxVector, o.avxVector));
    }

    DVector3 operator-(const DVector3 &o) const
    {
        return DVector3(_mm256_sub_pd(avxVector, o.avxVector));
    }

    DVector3 operator*(const DVector3 &o) const
    {
        return DVector3(_mm256_mul_pd(avxVector, o.avxVector));
    }

    DVector3 operator/(const DVector3 &o) const
    {
        return DVector3(_mm256_div_pd(avxVector, o.avxVector));
    }

#else
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
#endif

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