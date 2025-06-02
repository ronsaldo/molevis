#ifndef POTENTIALS_HPP
#define POTENTIALS_HPP

#pragma once

#include "cuda_runtime_api.h"

// Units simulationTimeStep
const double SimulationTimeStep = 1e-3f; // Picoseconds
const double BoltzmannConstantSI = 1.380649e-23; // m^2.K^-1
const double TargetTemperature = 10; // Kelvin

__host__ __device__ inline double quickPow6(double base)
{
    return (base*base)*(base*base)*(base*base);
}

__host__ __device__ inline double quickPow7(double base)
{
    return quickPow6(base)*base;
}

__host__ __device__ inline double quickPow12(double base)
{
    auto result = quickPow6(base);
    return result*result;
}

__host__ __device__ inline double quickPow13(double base)
{
    return quickPow12(base)*base;
}

inline double
lennardJonesPotential(double r, double sigma, double epsilon)
{
    return 4*epsilon*(quickPow12(sigma/r) - quickPow6(sigma/r));
}

inline __host__ __device__ double
lennardJonesDerivative(double r, double sigma, double epsilon)
{
    return 24*epsilon*(quickPow6(sigma)/quickPow7(r) - 2.0*quickPow12(sigma)/quickPow13(r));
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
hookPotential(double distance, double equilibriumDistance, double k)
{
    double delta = distance - equilibriumDistance;
    return 0.5*k * (delta*delta);
}

inline __host__ __device__ double
hookPotentialDerivative(double distance, double equilibriumDistance, double k)
{
    double delta = distance - equilibriumDistance;
    return k * delta;
}

#endif //POTENTIALS_HPP
