#ifndef MOLLEVIS_DAABOX_HPP
#define MOLLEVIS_DAABOX_HPP

#include "DVector3.hpp"
#include <math.h>
#include <algorithm>

struct DAABox
{
    static DAABox empty()
    {
        return DAABox{
            {INFINITY, INFINITY, INFINITY},
            {-INFINITY, -INFINITY, -INFINITY},
        };
    }

    void insertPoint(const DVector3 &p)
    {
        min.x = std::min(min.x, p.x);
        min.y = std::min(min.y, p.y);
        min.z = std::min(min.z, p.z);

        max.x = std::max(max.x, p.x);
        max.y = std::max(max.y, p.y);
        max.z = std::max(max.z, p.z);
    }

    DVector3 halfExtent()
    {
        return (max - min)*0.5;
    }

    DVector3 extent()
    {
        return max - min;
    }

    DVector3 center()
    {
        return min + halfExtent();
    }

    DVector3 min, max;
};


#endif //MOLLEVIS_AABOX_HPP
