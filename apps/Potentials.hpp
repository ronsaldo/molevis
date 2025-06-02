#ifndef POTENTIALS_HPP
#define POTENTIALS_HPP

#pragma once

#include "cuda_runtime_api.h"

// Units simulationTimeStep
const double SimulationTimeStep = 1e-3f; // Picoseconds
const double BoltzmannConstantSI = 1.380649e-23; // m^2.K^-1
const double TargetTemperature = 10; // Kelvin

__host__ __device__ inline double quickPow(double base, int exponent)
{
    if(exponent == 0)
        return 1;
    else if((exponent & 1) == 0)
        return quickPow(base*base, exponent>>1);
    else
        return base * quickPow(base*base, exponent>>1);
}

inline double
lennardJonesPotential(double r, double sigma, double epsilon)
{
    return 4*epsilon*(quickPow(sigma/r, 12) - quickPow(sigma/r, 6));
}

inline __host__ __device__ double
lennardJonesDerivative(double r, double sigma, double epsilon)
{
    return 24*epsilon*(quickPow(sigma, 6)/quickPow(r, 7) - 2.0*quickPow(sigma, 12)/quickPow(r, 13));
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
