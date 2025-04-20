#line 2

layout(local_size_x = 32) in;


float morsePotentialDerivative(float r, float D, float a, float re)
{
    float innerExp = exp(-a*(r - re));
    return -2.0*a*D*(1.0 - innerExp)*innerExp;
}

const uint TileSize = 32u;
shared vec3 tileAtomPositions[TileSize];
shared float tileAtomLennardJonesCutoff[TileSize];
shared float tileAtomLennardJonesEpsilon[TileSize];
shared float tileAtomLennardJonesSigma[TileSize];

shared uint tileBondFirstAtomIndex[TileSize];
shared uint tileBondSecondAtomIndex[TileSize];
shared vec3 tileBondFirstAtomPosition[TileSize];
shared float tileBondMorseEquilibriumDistance[TileSize];
shared float tileBondMorseWellDepth[TileSize];
shared float tileBondMorseWellWidth[TileSize];

void main()
{
    uint myAtomIndex = gl_GlobalInvocationID.x;
    vec3 myAtomPosition = AtomStateBuffer[myAtomIndex].position;
    vec3 myNetForce = AtomStateBuffer[myAtomIndex].netForce;
    float myLennardJonesCutoff = AtomDescriptionBuffer[myAtomIndex].lennardJonesCutoff;
    float myLennardJonesEpsilon = AtomDescriptionBuffer[myAtomIndex].lennardJonesEpsilon;
    float myLennardJonesSigma = AtomDescriptionBuffer[myAtomIndex].lennardJonesSigma;

    // Lennard-jones potential.
    for(uint firstAtomTileIndex = 0u; firstAtomTileIndex < atomCount; firstAtomTileIndex += TileSize)
    {
        uint fetchTileElementIndex = gl_LocalInvocationID.x;
        uint fetchAtomIndex = firstAtomTileIndex + fetchTileElementIndex;
        if(fetchAtomIndex < atomCount)
        {
            tileAtomPositions[fetchTileElementIndex] = AtomStateBuffer[fetchAtomIndex].position;
            tileAtomLennardJonesCutoff[fetchTileElementIndex] = AtomDescriptionBuffer[fetchAtomIndex].lennardJonesCutoff;
            tileAtomLennardJonesEpsilon[fetchTileElementIndex] = AtomDescriptionBuffer[fetchAtomIndex].lennardJonesEpsilon;
            tileAtomLennardJonesSigma[fetchTileElementIndex] = AtomDescriptionBuffer[fetchAtomIndex].lennardJonesSigma;
        }
        barrier();

        for(uint tileElementIndex = 0u; tileElementIndex < TileSize; ++tileElementIndex)
        {
            uint firstAtomIndex = firstAtomTileIndex + tileElementIndex;
        
            uint secondAtomIndex = myAtomIndex;
            vec3 secondPosition = myAtomPosition;

            if(firstAtomIndex == secondAtomIndex || firstAtomIndex >= atomCount || secondAtomIndex >= atomCount)
                continue;

            // Fetch the first position and the lennard jones coefficients.
            vec3 firstPosition = tileAtomPositions[tileElementIndex];

            float firstLennardJonesCutoff = tileAtomLennardJonesCutoff[tileElementIndex];
            float firstLennardJonesEpsilon = tileAtomLennardJonesEpsilon[tileElementIndex];
            float firstLennardJonesSigma = tileAtomLennardJonesEpsilon[tileElementIndex];

            float lennardJonesCutoff = max(firstLennardJonesCutoff, myLennardJonesCutoff);
            float lennardJonesEpsilon = sqrt(firstLennardJonesEpsilon*myLennardJonesEpsilon);
            float lennardJonesSigma = (firstLennardJonesSigma + myLennardJonesSigma) * 0.5;

            vec3 direction = secondPosition - firstPosition;
            float dist = length(direction);
            if(dist > 0.000001 /*&& dist < lennardJonesCutoff*/)
            {
                direction /= dist;
                vec3 force = -direction * lennardJonesDerivative(dist, lennardJonesSigma, lennardJonesEpsilon);
                myNetForce += force;
            }
        }

        barrier();
    }

    // Bond morse potential.
    /*
    for(uint bondTileIndex = 0u; bondTileIndex < bondCount; bondTileIndex += TileSize)
    {
        uint fetchTileElementIndex = gl_LocalInvocationID.x;
        uint fetchBondIndex = bondTileIndex + fetchTileElementIndex;
        if(fetchBondIndex < bondCount)
        {
            tileBondFirstAtomIndex[fetchTileElementIndex] = AtomBondDescriptionBuffer[fetchBondIndex].firstAtomIndex;
            tileBondSecondAtomIndex[fetchTileElementIndex] = AtomBondDescriptionBuffer[fetchBondIndex].secondAtomIndex;

            tileBondFirstAtomPosition[fetchTileElementIndex] = AtomStateBuffer[tileBondFirstAtomIndex[fetchTileElementIndex]].position;
            
            tileBondMorseEquilibriumDistance[fetchTileElementIndex] = AtomBondDescriptionBuffer[fetchBondIndex].morseEquilibriumDistance;
            tileBondMorseWellDepth[fetchTileElementIndex] = AtomBondDescriptionBuffer[fetchBondIndex].morseWellDepth;
            tileBondMorseWellWidth[fetchTileElementIndex] = AtomBondDescriptionBuffer[fetchBondIndex].morseWellWidth;
        }
        barrier();

        for(uint bondTileElementIndex = 0u; bondTileElementIndex < TileSize; ++bondTileElementIndex)
        {
            uint bondIndex = bondTileIndex + bondTileElementIndex;
            if(bondIndex >= bondCount || tileBondSecondAtomIndex[bondTileElementIndex] != myAtomIndex)
                continue;

            uint secondAtomIndex = myAtomIndex;
            vec3 secondPosition = myAtomPosition;

            uint firstAtomIndex = tileBondFirstAtomIndex[bondTileElementIndex];
            vec3 firstPosition = tileBondFirstAtomPosition[bondTileElementIndex];

            float morseEquilibriumDistance = tileBondMorseEquilibriumDistance[bondTileElementIndex];
            float morseWellDepth = tileBondMorseWellDepth[bondTileElementIndex];
            float morseWellWidth = tileBondMorseWellWidth[bondTileElementIndex];

            vec3 direction = secondPosition - firstPosition;
            float dist = length(direction);
            if(dist > 0.000001)
            {
                direction /= dist;

                vec3 force = -direction * morsePotentialDerivative(dist, morseWellDepth, morseWellWidth, morseEquilibriumDistance);
                myNetForce += force;
            }
        }

        barrier();
    }*/

    AtomStateBuffer[myAtomIndex].netForce = myNetForce;
}
