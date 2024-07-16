#ifndef MOLLEVIS_MATRIX4x4_HPP
#define MOLLEVIS_MATRIX4x4_HPP

#include "Vector4.hpp"
#include "Matrix3x3.hpp"

struct Matrix4x4
{
    static Matrix4x4 perspective(float fovy, float aspect, float nearDistance, float farDistance, bool flipVertically)
    {
        float fovyRad = fovy *0.5f * (M_PI / 180.0f);
        float top = nearDistance * tan(fovyRad);
        float right = top * aspect;

        return frustum(-right, right, -top, top, nearDistance, farDistance, flipVertically);
    }

    static Matrix4x4 frustum(float left, float right, float bottom, float top, float near, float far, bool flipVertically)
    {
        float flipYFactor = flipVertically ? -1.0f : 1.0f;
        return Matrix4x4{{
            {2*near / (right - left), 0, 0, 0},
            {0, flipYFactor*2*near / (top - bottom), 0, 0},
            {(right + left) / (right - left), flipYFactor*(top + bottom) / (top - bottom), -far / (far - near), -1},
            {0, 0, -near*far / (far - near), 0}
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

    Matrix4x4 operator+(const Matrix4x4 &o) const
    {
        return Matrix4x4{{columns[0] + o.columns[0], columns[1] + o.columns[1], columns[2] + o.columns[2], columns[3] + o.columns[3]}};
    }

    Matrix4x4 operator-(const Matrix4x4 &o) const
    {
        return Matrix4x4{{columns[0] - o.columns[0], columns[1] - o.columns[1], columns[2] - o.columns[2], columns[3] - o.columns[3]}};
    }

    Vector4 columns[4];
};

#endif //MOLLEVIS_MATRIX4x4_HPP
