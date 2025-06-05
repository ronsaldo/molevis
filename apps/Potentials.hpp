#ifndef POTENTIALS_HPP
#define POTENTIALS_HPP

#pragma once

#include "cuda_runtime_api.h"

// Units simulationTimeStep
const double SimulationTimeStep = 1e-3f; // Picoseconds
const double BoltzmannConstant = 0.8314459920816467; // Constant from https://github.com/PolymerTheory/MDFromScratch/blob/main/MDFS.ipynb
const double AvogadroNumber = 6.0221409e+26; // Constant from https://github.com/PolymerTheory/MDFromScratch/blob/main/MDFS.ipynb
const double TargetTemperature = 10; // Kelvin

__host__ __device__ inline float averageKineticEnergyToTemperatureFloat(float kineticEnergy)
{
    return (2.0/3.0) * kineticEnergy / float(BoltzmannConstant);
}

__host__ __device__ inline float quickFPow6(float base)
{
    return (base*base)*(base*base)*(base*base);
}

__host__ __device__ inline float quickFPow7(float base)
{
    return quickFPow6(base)*base;
}

__host__ __device__ inline float quickFPow12(float base)
{
    auto result = quickFPow6(base);
    return result*result;
}

__host__ __device__ inline float quickFPow13(float base)
{
    return quickFPow12(base)*base;
}

__host__ __device__ inline double averageKineticEnergyToTemperatureDouble(double kineticEnergy)
{
    return (2.0/3.0) * kineticEnergy / BoltzmannConstant;
}

__host__ __device__ inline double quickDPow6(double base)
{
    return (base*base)*(base*base)*(base*base);
}

__host__ __device__ inline double quickDPow7(double base)
{
    return quickDPow6(base)*base;
}

__host__ __device__ inline double quickDPow12(double base)
{
    auto result = quickDPow6(base);
    return result*result;
}

__host__ __device__ inline double quickDPow13(double base)
{
    return quickDPow12(base)*base;
}

inline double
lennardJonesDoublePotential(double r, double sigma, double epsilon)
{
    return 4*epsilon*(quickDPow12(sigma/r) - quickDPow6(sigma/r));
}

inline __host__ __device__ double
lennardJonesDoubleDerivative(double r, double sigma, double epsilon)
{
    return 24*epsilon*(quickDPow6(sigma)/quickDPow7(r) - 2.0*quickDPow12(sigma)/quickDPow13(r));
}

inline float
lennardJonesSinglePotential(float r, float sigma, float epsilon)
{
    return 4*epsilon*(quickFPow12(sigma/r) - quickFPow6(sigma/r));
}

inline __host__ __device__ float
lennardJonesSingleDerivative(float r, float sigma, float epsilon)
{
    return 24*epsilon*(quickFPow6(sigma)/quickFPow7(r) - 2.0*quickFPow12(sigma)/quickFPow13(r));
}


inline double
morsePotential(double r, double De, double a, double re)
{
    double interior = (1 - exp(-a*(r - re)));
    return De*interior*interior;
}

inline double
morsePotentialDerivative(double r, double De, double a, double re)
{
    double innerExp = exp(-a*(r - re));
    return -2.0*a*De*(1.0 - innerExp)*innerExp;
}

inline double
hookPotentialDouble(double distance, double equilibriumDistance, double k)
{
    double delta = distance - equilibriumDistance;
    return 0.5*k * (delta*delta);
}

inline __host__ __device__ double
hookPotentialDoubleDerivative(double distance, double equilibriumDistance, double k)
{
    double delta = distance - equilibriumDistance;
    return k * delta;
}

inline double
hookPotentialSingle(float distance, float equilibriumDistance, float k)
{
    float delta = distance - equilibriumDistance;
    return 0.5*k * (delta*delta);
}

inline __host__ __device__ float
hookPotentialSingleDerivative(float distance, float equilibriumDistance, float k)
{
    float delta = distance - equilibriumDistance;
    return k * delta;
}

#endif //POTENTIALS_HPP
