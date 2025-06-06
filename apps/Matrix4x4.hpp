#ifndef MOLLEVIS_MATRIX4x4_HPP
#define MOLLEVIS_MATRIX4x4_HPP

#include "Vector4.hpp"
#include "Matrix3x3.hpp"
#include "AGPU/agpu.hpp"
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct Matrix4x4
{
    static Matrix4x4 fromAgpu(const agpu_matrix4x4f &am)
    {
        return Matrix4x4{{
            {am.c1.x, am.c1.y, am.c1.z, am.c1.w},
            {am.c2.x, am.c2.y, am.c2.z, am.c2.w},
            {am.c3.x, am.c3.y, am.c3.z, am.c3.w},
            {am.c4.x, am.c4.y, am.c4.z, am.c4.w},
        }};
    }

    static Matrix4x4 withScaleFactor(float scale)
    {
        return Matrix4x4{{
            {scale, 0, 0, 0},
            {0, scale, 0, 0},
            {0, 0, scale, 0},
            {0, 0, 0, 1},
        }};
    }

    static Matrix4x4 identity()
    {
        return Matrix4x4{{
            {1, 0, 0, 0},
            {0, 1, 0, 0},
            {0, 0, 1, 0},
            {0, 0, 0, 1},
        }};
    }

    static Matrix4x4 perspective(float fovy, float aspect, float nearDistanceDistance, float farDistanceDistance, bool flipVertically)
    {
        float fovyRad = float(fovy *0.5f * (M_PI / 180.0f));
        float top = float(nearDistanceDistance * tan(fovyRad));
        float right = top * aspect;

        return frustum(-right, right, -top, top, nearDistanceDistance, farDistanceDistance, flipVertically);
    }

    static Matrix4x4 frustum(float left, float right, float bottom, float top, float nearDistance, float farDistance, bool flipVertically)
    {
        float flipYFactor = flipVertically ? -1.0f : 1.0f;
        return Matrix4x4{{
            {2*nearDistance / (right - left), 0, 0, 0},
            {0, flipYFactor*2*nearDistance / (top - bottom), 0, 0},
            {(right + left) / (right - left), flipYFactor*(top + bottom) / (top - bottom), nearDistance / (farDistance - nearDistance), -1},
            {0, 0, nearDistance*farDistance / (farDistance - nearDistance), 0}
        }};
    }

    static Matrix4x4 withMatrix3x3AndTranslation(const Matrix3x3 &mat, const Vector3 &translation)
    {
        return Matrix4x4{
            mat.columns[0].asVector4(),
            mat.columns[1].asVector4(),
            mat.columns[2].asVector4(),
            Vector4{translation.x, translation.y, translation.z, 1.0f}
        };
    }

    Matrix4x4 transposed() const
    {
        return Matrix4x4{
            Vector4{columns[0].x, columns[1].x, columns[2].x, columns[3].x},
            Vector4{columns[0].y, columns[1].y, columns[2].y, columns[3].y},
            Vector4{columns[0].z, columns[1].z, columns[2].z, columns[3].z},
            Vector4{columns[0].w, columns[1].w, columns[2].w, columns[3].w},
        };
    }

    Matrix4x4 operator+(const Matrix4x4 &o) const
    {
        return Matrix4x4{{columns[0] + o.columns[0], columns[1] + o.columns[1], columns[2] + o.columns[2], columns[3] + o.columns[3]}};
    }

    Matrix4x4 operator-(const Matrix4x4 &o) const
    {
        return Matrix4x4{{columns[0] - o.columns[0], columns[1] - o.columns[1], columns[2] - o.columns[2], columns[3] - o.columns[3]}};
    }
    
    Vector4 operator*(const Vector4 &v) const
    {
        return columns[0]*v.x + columns[1]*v.y + columns[2]*v.z + columns[3]*v.w;
    }

    Matrix4x4 operator*(const Matrix4x4 &o) const
    {
        return Matrix4x4{{
            columns[0]*o.columns[0].x + columns[1]*o.columns[0].y + columns[2]*o.columns[0].z + columns[3]*o.columns[0].w,
            columns[0]*o.columns[1].x + columns[1]*o.columns[1].y + columns[2]*o.columns[1].z + columns[3]*o.columns[1].w,
            columns[0]*o.columns[2].x + columns[1]*o.columns[2].y + columns[2]*o.columns[2].z + columns[3]*o.columns[2].w,
            columns[0]*o.columns[3].x + columns[1]*o.columns[3].y + columns[2]*o.columns[3].z + columns[3]*o.columns[3].w
        }};
    }

    Matrix3x3 minorMatrixAt(int row, int column) const
    {
        switch(column)
        {
        case 0: return Matrix3x3{{
                columns[1].minorAt(row),
                columns[2].minorAt(row),
                columns[3].minorAt(row),
            }};
        case 1: return Matrix3x3{{
                columns[0].minorAt(row),
                columns[2].minorAt(row),
                columns[3].minorAt(row),
            }};
        case 2: return Matrix3x3{{
                columns[0].minorAt(row),
                columns[1].minorAt(row),
                columns[3].minorAt(row),
            }};
        case 3: return Matrix3x3{{
                columns[0].minorAt(row),
                columns[1].minorAt(row),
                columns[2].minorAt(row),
            }};
        default: abort();
        }
    }
    
    float minorAt(int row, int column) const
    {
        return minorMatrixAt(row, column).determinant();
    }

    float determinant() const
    {
	    return minorAt(0, 0)*columns[0].x
             - minorAt(0, 1)*columns[1].x
             + minorAt(0, 2)*columns[2].x
             - minorAt(0, 3)*columns[3].x;
    }

    float cofactorAt(int row, int column) const
    {
        return minorMatrixAt(row, column).determinant() * ((row + column) & 1 ? -1 : 1);
    }

    Matrix4x4 adjugate() const
    {
        return Matrix4x4{{
            Vector4{
                cofactorAt(0, 0),
                cofactorAt(0, 1),
                cofactorAt(0, 2),
                cofactorAt(0, 3)
            },
            Vector4{
                cofactorAt(1, 0),
                cofactorAt(1, 1),
                cofactorAt(1, 2),
                cofactorAt(1, 3)
            },
            Vector4{
                cofactorAt(2, 0),
                cofactorAt(2, 1),
                cofactorAt(2, 2),
                cofactorAt(2, 3)
            },
            Vector4{
                cofactorAt(3, 0),
                cofactorAt(3, 1),
                cofactorAt(3, 2),
                cofactorAt(3, 3)
            },
        }};
    }

    Matrix4x4 inverse() const
    {
        auto det = determinant();
        auto adj = adjugate();
        return Matrix4x4{{
            adj.columns[0] / det,
            adj.columns[1] / det,
            adj.columns[2] / det,
            adj.columns[3] / det,
        }};
    }

    Vector4 columns[4];
};

#endif //MOLLEVIS_MATRIX4x4_HPP
