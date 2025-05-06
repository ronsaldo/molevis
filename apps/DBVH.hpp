#ifndef MOLEVIS_DBVH_HPP
#define MOLEVIS_DBVH_HPP

#include "DAABox.hpp"
#include "DSphere.hpp"
#include <vector>

struct DBVHNode
{
    bool isLeaf;
    DAABox boundingBox;
    union
    {
        struct
        {
            uint32_t leftChild;
            uint32_t rightChild;
        };
        struct
        {
            uint32_t atomIndex;
            uint64_t quantizedCenterPosition;
        };
    };
};

struct DBVH
{
    void swap(DBVH &other)
    {
        nodes.swap(other.nodes);
    }

    template<typename FT>
    void atomsIntersectingSphereDo(const DSphere &sphere, size_t nodeIndex, const FT &function)
    {
        const auto &node = nodes[nodeIndex];
        if(!node.boundingBox.intersectsSphere(sphere))
            return;

        if(node.isLeaf)
        {
            function(nodeIndex);
        }
        else
        {
            atomsIntersectingSphereDo(sphere, node.leftChild, function);
            atomsIntersectingSphereDo(sphere, node.rightChild, function);
        }
    }

    template<typename FT>
    void atomsIntersectingSphereDo(const DSphere &sphere, const FT &function)
    {
        if(nodes.empty())
            return;

        atomsIntersectingSphereDo(sphere, nodes.size() - 1, function);
    }

    void buildInnerNodes()
    {
        uint32_t sourceIndex = 0;
        while(sourceIndex + 1 < nodes.size())
        {
            uint32_t leftNodeIndex = sourceIndex;
            uint32_t rightNodeIndex = sourceIndex + 1;
            auto &leftNode = nodes[leftNodeIndex];
            auto &rightNode = nodes[rightNodeIndex];

            DBVHNode innerNode;
            innerNode.isLeaf = false;
            innerNode.boundingBox = leftNode.boundingBox.unionWith(rightNode.boundingBox);
            innerNode.leftChild = leftNodeIndex;
            innerNode.rightChild = rightNodeIndex;
            
            nodes.push_back(innerNode);
            sourceIndex += 2;
        }
    }

    std::vector<DBVHNode> nodes;
};

inline uint64_t computeZOrder(uint64_t x, uint64_t y, uint64_t z)
{
    uint64_t result = 0;
    uint64_t destIndex = 0;
    for(int i = 0; i < 16; ++i)
    {
        result |= (x & (1<<i)) << destIndex++;
        result |= (y & (1<<i)) << destIndex++;
        result |= (z & (1<<i)) << destIndex++;
    }

    return result;
}
#endif // MOLEVIS_DBVH_HPP
