#ifndef MOLLEVIS_VECTOR2_HPP
#define MOLLEVIS_VECTOR2_HPP

struct alignas(8) Vector2
{
    float x, y;

    float dot(const Vector2 &o) const
    {
        return x*o.x + y*o.y;
    }

    Vector2 operator+(const Vector2 &o) const
    {
        return Vector2{x + o.x, y + o.y};
    }

    Vector2 operator-(const Vector2 &o) const
    {
        return Vector2{x - o.x, y - o.y};
    }

    Vector2 operator*(const Vector2 &o) const
    {
        return Vector2{x * o.x, y * o.y};
    }

    Vector2 operator/(const Vector2 &o) const
    {
        return Vector2{x / o.x, y / o.y};
    }

    Vector2 operator+=(const Vector2 &o)
    {
        return *this = *this + o;
    }

};

#endif //MOLLEVIS_VECTOR2_HPP