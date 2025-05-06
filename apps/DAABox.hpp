#ifndef MOLLEVIS_DAABOX_HPP
#define MOLLEVIS_DAABOX_HPP

#include "DVector3.hpp"
#include "DSphere.hpp"
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

    static DAABox withCenterAndHalfExtent(const DVector3 &center, const DVector3 &halfExtent)
    {
        return DAABox{
            center - halfExtent,
            center + halfExtent
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

    void insertBox(const DAABox &b)
    {
        min.x = std::min(min.x, b.min.x);
        min.y = std::min(min.y, b.min.y);
        min.z = std::min(min.z, b.min.z);

        max.x = std::max(max.x, b.max.x);
        max.y = std::max(max.y, b.max.y);
        max.z = std::max(max.z, b.max.z);
    }

    DAABox unionWith(const DAABox &o) const
    {
        auto result = *this;
        result.insertBox(o);
        return result;
    }

    DVector3 halfExtent() const
    {
        return (max - min)*0.5;
    }

    DVector3 extent() const
    {
        return max - min;
    }

    DVector3 center() const
    {
        return min + halfExtent();
    }

    double distanceSquaredForPoint(const DVector3 &point) const
    {
        return ((point - center()).abs() - halfExtent()).max(DVector3(0, 0, 0)).length2();
    }

    bool intersectsSphere(const DSphere &sphere) const
    {
        return distanceSquaredForPoint(sphere.center) <= sphere.radius*sphere.radius;
    }

    DVector3 min, max;
};


#endif //MOLLEVIS_AABOX_HPP
