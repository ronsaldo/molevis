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
    float padding;

    Matrix4x4 projectionMatrix;
    Matrix4x4 inverseProjectionMatrix;
    Matrix4x4 viewMatrix;
    Matrix4x4 inverseViewMatrix;
    
    Matrix4x4 atomModelMatrix;
    Matrix4x4 atomInverseModelMatrix;
};

#endif //MOLLEVIS_CAMERA_STATE_HPP
