#ifndef LENNARD_JONES_TABLE_HPP
#define LENNARD_JONES_TABLE_HPP

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include "PeriodicTable.hpp"

struct LennardJonesTableEntry
{
    float sigma;
    float epsilon;
};

class LennardJonesTable
{
public:
    static constexpr int MaxAtomicNumber = 256;

    bool loadFromFile(const std::string &filename, PeriodicTable &periodicTable)
    {
        std::ifstream in(filename);
        if(!in.good())
        {
            fprintf(stderr, "Failed to open the lennad-jones coefficient table\n");
            return false;
        }

        std::string line;
        for(std::getline(in, line); in.good(); std::getline(in, line))
        {
            parseLine(line, periodicTable);
        }

        return true;
    }

    void parseLine(std::string line, PeriodicTable &periodicTable)
    {
        if(line.empty() || line[0] == '#')
            return;

        std::istringstream in(line);
        std::string firstSymbol, secondSymbol;
        double cutoff, epsilon, sigma;
        in >> firstSymbol >> secondSymbol >> cutoff >> epsilon >> sigma;

        if(firstSymbol != secondSymbol)
            return;

        auto it = periodicTable.atomicSymbolMap.find(firstSymbol);
        if(it == periodicTable.atomicSymbolMap.end())
            return;

        auto atomicNumber = it->second;

        auto &atomDescription = periodicTable.elements[atomicNumber];
        atomDescription.lennardJonesCutoff = cutoff;
        atomDescription.lennardJonesEpsilon = epsilon;
        atomDescription.lennardJonesSigma = sigma;
    }


};

#endif //LENNARD_JONES_TABLE_HPP