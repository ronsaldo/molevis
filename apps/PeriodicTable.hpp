#ifndef MOLEVIS_PERIODIC_TABLE_HPP
#define MOLEVIS_PERIODIC_TABLE_HPP

#pragma once

#include "AtomDescription.hpp"
#include <stdio.h>
#include <istream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <assert.h>

/**
 * Dataset source: https://gist.github.com/GoodmanSciences/c2dd862cd38f21b0ad36b8f96b4bf1ee
 */
struct PeriodicTableRecord
{
    int atomicNumber = 0;
    std::string name;
    std::string symbol;
    double atomicMass = 1;
    int numberOfNeutrons = 0;
    int numberOfProtons = 0;
    int numberOfElectrons = 0;
    int period = 0;
    int group = 0;
    double atomicRadius = 1.0f;
    double lennardJonesCutoff = 1.0f;
    double lennardJonesEpsilon = 1.0f;
    double lennardJonesSigma = 1.0f;
};

inline std::vector<std::string> tokenizeCSV(const std::string &line, char delimiter = ',')
{
    std::vector<std::string> tokens;
    std::string currentToken;
    for(size_t i = 0; i < line.size(); ++i)
    {
        auto c = line[i];
        if(c == delimiter)
        {
            tokens.push_back(currentToken);
            currentToken.clear();
        }
        else
        {
            currentToken.push_back(c);
        }

    }

    return tokens;
}

struct PeriodicTable
{
    PeriodicTableRecord *getElementRecordForSymbol(const std::string &symbol)
    {
        auto it = atomicSymbolMap.find(symbol);
        if(it != atomicSymbolMap.end())
            return elements.data() + it->second;

        return nullptr;
    }

    AtomDescription makeAtomDescriptionForSymbol(const std::string &symbol)
    {
        AtomDescription desc = {};
        auto record = getElementRecordForSymbol(symbol);
        if(record)
        {
            desc.atomNumber = record->atomicNumber;
            desc.radius = float(record->atomicRadius);
            desc.mass = float(record->atomicMass);

            desc.lennardJonesCutoff = float(record->lennardJonesCutoff);
            desc.lennardJonesEpsilon = float(record->lennardJonesEpsilon);
            desc.lennardJonesSigma = float(record->lennardJonesSigma);
        }

        return desc;
    }

    bool loadFromFile(const std::string &filename)
    {
        std::ifstream in("assets/datasets/periodic-table-of-elements.csv");
        if(!in.good())
        {
            fprintf(stderr, "Failed to open the periodic table dataset\n");
            return false;
        }

        // Empty record.
        elements.push_back(PeriodicTableRecord{});

        std::string header;
        std::getline(in, header);
        auto headerNames = tokenizeCSV(header);

        std::string line;
        for(std::getline(in, line); !line.empty(); std::getline(in, line))
        {
            auto tokens = tokenizeCSV(line);
            PeriodicTableRecord record;
            record.atomicNumber = atoi(tokens[0].c_str());
            record.name = tokens[1];
            record.symbol = tokens[2];

            record.atomicMass = atof(tokens[3].c_str());
            record.numberOfNeutrons = atoi(tokens[4].c_str());
            record.numberOfProtons = atoi(tokens[5].c_str());
            record.numberOfElectrons = atoi(tokens[6].c_str());
            record.period = atoi(tokens[7].c_str());
            record.group = atoi(tokens[8].c_str());

            record.atomicRadius = atof(tokens[16].c_str());

            assert(size_t(record.atomicNumber) == elements.size());
            elements.push_back(record);
        }

        for(size_t i = 0; i < elements.size(); ++i)
        {
            auto &element = elements[i];
            atomicSymbolMap.insert(std::make_pair(element.symbol, i));
        }
        
        return true;
    }

    std::vector<PeriodicTableRecord> elements;
    std::unordered_map<std::string, size_t> atomicSymbolMap;
};

#endif //MOLEVIS_PERIODIC_TABLE_HPP
