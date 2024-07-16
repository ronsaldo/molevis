#ifndef MOLLEVIS_MATRIX3x3_HPP
#define MOLLEVIS_MATRIX3x3_HPP

#include "Vector3.hpp"
#include <math.h>

struct Matrix3x3
{
    static Matrix3x3 identity()
    {
        return Matrix3x3{
            Vector3{1, 0, 0},
            Vector3{0, 1, 0},
            Vector3{0, 0, 1},
        };
    }

    static Matrix3x3 XRotation(float angle)
    {
        float c = cos(angle);
        float s = sin(angle);
        return Matrix3x3{{
            {1, 0, 0},
            {0, c, s},
            {0, -s, c}
        }};
    }

    static Matrix3x3 YRotation(float angle)
    {
        float c = cos(angle);
        float s = sin(angle);
        return Matrix3x3{{
            {c, 0, -s},
            {0, 1, 0},
            {s, 0, c}
        }};
    }

    Matrix3x3 transposed() const
    {
        return Matrix3x3{
            Vector3{columns[0].x, columns[1].x, columns[2].x},
            Vector3{columns[0].y, columns[1].y, columns[2].y},
            Vector3{columns[0].z, columns[1].z, columns[2].z}
        };
    }

    Matrix3x3 operator+(const Matrix3x3 &o) const
    {
        return Matrix3x3{{columns[0] + o.columns[0], columns[1] + o.columns[1], columns[2] + o.columns[2]}};
    }

    Matrix3x3 operator-(const Matrix3x3 &o) const
    {
        return Matrix3x3{{columns[0] - o.columns[0], columns[1] - o.columns[1], columns[2] - o.columns[2]}};
    }

    Vector3 operator*(const Vector3 &v) const
    {
        return columns[0]*v.x + columns[1]*v.y + columns[2]*v.z;
    }

    Matrix3x3 operator*(const Matrix3x3 &o) const
    {
        Matrix3x3 s = transposed();
        return Matrix3x3{{
            Vector3{
                s.columns[0].dot(o.columns[0]),
                s.columns[0].dot(o.columns[1]),
                s.columns[0].dot(o.columns[2])
            },
            Vector3{
                s.columns[1].dot(o.columns[0]),
                s.columns[1].dot(o.columns[1]),
                s.columns[1].dot(o.columns[2])
            },
            Vector3{
                s.columns[2].dot(o.columns[0]),
                s.columns[2].dot(o.columns[1]),
                s.columns[2].dot(o.columns[2])
            },
        }};
    }

    Vector3 columns[3];
};

#endif //MOLLEVIS_MATRIX3x3_HPP
