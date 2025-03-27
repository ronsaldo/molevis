#ifndef MOLLEVIS_AABOX_HPP
#define MOLLEVIS_AABOX_HPP

#include "Vector3.hpp"
#include <math.h>
#include <algorithm>

struct AABox
{
    static AABox empty()
    {
        return AABox{
            {INFINITY, INFINITY, INFINITY},
            {-INFINITY, -INFINITY, -INFINITY},
        };
    }

    void insertPoint(const Vector3 &p)
    {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        min.z = std::min(min.z, p.z);

        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        max.z = std::max(max.z, p.z);
    }

    Vector3 halfExtent()
    {
        return (max - min)*0.5;
    }

    Vector3 center()
    {
        return min + halfExtent();
    }

    Vector3 min, max;
};


#endif //MOLLEVIS_AABOX_HPP
