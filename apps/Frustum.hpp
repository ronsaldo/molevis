#ifndef FRUSTUM_HPP
#define FRUSTUM_HPP

#include "Vector2.hpp"
#include "Vector3.hpp"
#include "Matrix4x4.hpp"
#include "AABox.hpp"
#include "Plane.hpp"
#include "Ray.hpp"

template<typename T, typename S>
T mix(const T &a, const T &b, const S &lambda)
{
    return a*S(1.0-lambda) + b*lambda;
}

struct Frustum
{
	Vector3 leftBottomNear;
	Vector3 rightBottomNear;
	Vector3 leftTopNear;
	Vector3 rightTopNear;
	Vector3 leftBottomFar;
	Vector3 rightBottomFar;
	Vector3 leftTopFar;
	Vector3 rightTopFar;

	AABox boundingBox;

	template <typename FT>
	void cornersDo(const FT &function)
	{
		function(leftBottomNear);
		function(rightBottomNear);
		function(leftTopNear);
		function(rightTopNear);
		function(leftBottomFar);
		function(rightBottomFar);
		function(leftTopFar);
		function(rightTopFar);
	}

	void makeOrtho(float left, float right, float bottom, float top, float nearDistance, float farDistance);
	void makeFrustum(float left, float right, float bottom, float top, float nearDistance, float farDistance);
	void makePerspective(float fovy, float aspect, float nearDistance, float farDistance);
	void computePlanes();
	void computeBoundingBox();

    template<typename FT>
    Frustum transformedWithFunction(const FT &function)
    {
        Frustum transformed;
        transformed.leftBottomNear  = function(leftBottomNear);
        transformed.rightBottomNear = function(rightBottomNear);
        transformed.leftTopNear     = function(leftTopNear);
        transformed.rightTopNear    = function(rightTopNear);
        transformed.leftBottomFar   = function(leftBottomFar);
        transformed.rightBottomFar  = function(rightBottomFar);
        transformed.leftTopFar      = function(leftTopFar);
        transformed.rightTopFar     = function(rightTopFar);
        transformed.computePlanes();
        transformed.computeBoundingBox();
        return transformed;
    }

    Vector3 normalizedPointInNearPlane(const Vector2 &normalizedPoint)
    {
        return mix(mix(leftBottomNear, rightBottomNear, normalizedPoint.x),
            mix(leftTopNear, rightTopNear, normalizedPoint.x),
            normalizedPoint.y);
    }

    Vector3 normalizedPointInFarPlane(const Vector2 &normalizedPoint)
    {
        return mix(mix(leftBottomFar, rightBottomFar, normalizedPoint.x),
            mix(leftTopFar, rightTopFar, normalizedPoint.x),
            normalizedPoint.y);        
    }

    Ray rayForNormalizedPoint(const Vector2 &normalizedPoint)
    {
        auto nearPoint = normalizedPointInNearPlane(normalizedPoint);
        auto farPoint = normalizedPointInFarPlane(normalizedPoint);
        return Ray::fromTo(nearPoint, farPoint);
    }

	Frustum splitAtNearAndFarLambda(float nearLambda, float farLambda);
	Frustum transformedWith(const Matrix4x4 &transform);
    
    Plane planes[6];
};

#endif //UPHYSICS_FRUSTUM_HPP