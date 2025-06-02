#ifndef MOLLEVIS_DDVector2_HPP
#define MOLLEVIS_DDVector2_HPP

struct alignas(16) DVector2
{
    double x, y;

    DVector2()
        : x(0), y(0) {}

    DVector2(double s)
        : x(s), y(s) {}

    DVector2(double cx, double cy)
        : x(cx), y(cy) {}

    double dot(const DVector2 &o) const
    {
        return x*o.x + y*o.y;
    }

    DVector2 operator+(const DVector2 &o) const
    {
        return DVector2{x + o.x, y + o.y};
    }

    DVector2 operator-(const DVector2 &o) const
    {
        return DVector2{x - o.x, y - o.y};
    }

    DVector2 operator*(const DVector2 &o) const
    {
        return DVector2{x * o.x, y * o.y};
    }

    DVector2 operator/(const DVector2 &o) const
    {
        return DVector2{x / o.x, y / o.y};
    }

    DVector2 operator+=(const DVector2 &o)
    {
        return *this = *this + o;
    }

};

#endif //MOLLEVIS_DVector2_HPP