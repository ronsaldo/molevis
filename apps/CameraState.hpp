#ifndef MOLLEVIS_CAMERA_STATE_HPP
#define MOLLEVIS_CAMERA_STATE_HPP

#include <stdint.h>
#include "Matrix4x4.hpp"

struct CameraState
{
    uint32_t screenWidth = 640;
    uint32_t screenHeight = 480;

    uint32_t flipVertically = false;
    float nearDistance = 0.1f;
    float farDistance = 1000.0f;

    Matrix4x4 projectionMatrix;
    Matrix4x4 viewMatrix;
    Matrix4x4 inverseViewMatrix;
};

#endif //MOLLEVIS_CAMERA_STATE_HPP
