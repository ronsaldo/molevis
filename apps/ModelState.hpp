#ifndef MOLLEVIS_MODEL_STATE_HPP
#define MOLLEVIS_MODEL_STATE_HPP

#include <stdint.h>
#include "Matrix4x4.hpp"

struct ModelState
{
    Matrix4x4 modelMatrix;
    Matrix4x4 inverseModelMatrix;
};

#endif //MOLLEVIS_CAMERA_STATE_HPP
