#ifndef MOLLEVIS_PUSH_CONSTANTS_HPP
#define MOLLEVIS_PUSH_CONSTANTS_HPP

#include <stdint.h>

struct PushConstants
{
    float timeStep;
    uint32_t atomCount;
    uint32_t bondCount;
};

#endif //MOLLEVIS_PUSH_CONSTANTS_HPP