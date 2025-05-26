#ifndef POTENTIALS_HPP
#define POTENTIALS_HPP

#pragma once

#include "cuda_runtime_api.h"

// Units simulationTimeStep
const double SimulationTimeStep = 1e-3f; // Picoseconds
const double BoltzmannConstantSI = 1.380649e-23; // m^2.K^-1
const double TargetTemperature = 10; // Kelvin

inline double
lennardJonesPotential(double r, double sigma, double epsilon)
{
    return 4*epsilon*(pow(sigma/r, 12) - pow(sigma/r, 6));
}

inline __host__ __device__ double
lennardJonesDerivative(double r, double sigma, double epsilon)
{
    return 24*epsilon*(pow(sigma, 6)/pow(r, 7) - 2.0*pow(sigma, 12)/pow(r, 13));
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

inline double
hookPotentialDerivative(double distance, double equilibriumDistance, double k)
{
    double delta = distance - equilibriumDistance;
    return k * delta;
}

#endif //POTENTIALS_HPP
