#ifndef PDB_FORMAT_HPP
#define PDB_FORMAT_HPP

#include <string>
#include <fstream>
#include <vector>

struct PDBFileAtom
{
    int serial;
    std::string name;
    char altLoc;
    std::string residueName;
    char chainID;
    int resSeq;
    char iCode;
    float x;
    float y;
    float z;
    float occupancy;
    float tempFactor;
    std::string element;
    std::string charge;
};

class PDBFile
{
public:
    PDBFile()
    {
    }

    void parseHeaderRecord(std::string line)
    {

    }

    void parseTitleRecord(std::string line)
    {

    }

    void parseAuthorRecord(std::string line)
    {

    }

    void parseCompoundRecord(std::string line)
    {

    }

    void parseSourceRecord(std::string line)
    {

    }

    void parseKeywordsRecord(std::string line)
    {

    }

    void parseExpDataRecord(std::string line)
    {

    }

    void parseRevDataRecord(std::string line)
    {

    }

    void parseJournalRecord(std::string line)
    {

    }
    
    void parseRemarkRecord(std::string line)
    {

    }

    void parseDBRef(std::string line)
    {

    }

    void parseSequenceAdv(std::string line)
    {

    }

    void parseSequenceRes(std::string line)
    {

    }

    void parseFormula(std::string line)
    {

    }

    void parseHelix(std::string line)
    {

    }

    void parseSSBond(std::string line)
    {
    }

    void parseCryst1(std::string line)
    {
    }

    void parseOriginX1(std::string line)
    {
    }

    void parseOriginX2(std::string line)
    {
    }

    void parseOriginX3(std::string line)
    {
    }

    void parseScale1(std::string line)
    {
    }

    void parseScale2(std::string line)
    {
    }

    void parseScale3(std::string line)
    {
    }

    void parseAtom(std::string line)
    {
        PDBFileAtom atom;
        atom.serial = atoi(line.substr(6, 11).c_str());
        atom.name = line.substr(12, 16);
        atom.altLoc = line[16];
        atom.residueName = line.substr(17, 20);
        atom.chainID = line[21];
        atom.resSeq = atoi(line.substr(22, 26).c_str());
        atom.iCode = line[26];
        atom.x = atof(line.substr(30, 38).c_str());
        atom.y = atof(line.substr(38, 46).c_str());
        atom.z = atof(line.substr(46, 54).c_str());
        atom.occupancy = atof(line.substr(54, 60).c_str());
        atom.tempFactor = atof(line.substr(60, 66).c_str());
        atom.element = line.substr(76, 78);
        atom.charge = line.substr(78, 80);

        atoms.push_back(atom);
    }

    void parseTer(std::string line)
    {
    }

    void parseHetAtom(std::string line)
    {
    }

    void parseConnection(std::string line)
    {
    }

    void parseMaster(std::string line)
    {
    }

    void parseEnd(std::string line)
    {
    }

    void openAndParsePDBFile(const std::string &filename)
    {
        std::ifstream inputStream(filename);
        while(inputStream.good())
        {
            std::string line;
            std::getline(inputStream, line);
            auto recordType = line.substr(0, 6);
            if(recordType == "HEADER") parseHeaderRecord(line);
            else if (recordType == "TITLE ") parseTitleRecord(line);
            else if (recordType == "COMPND") parseCompoundRecord(line);
            else if (recordType == "SOURCE") parseSourceRecord(line);
            else if (recordType == "KEYWDS") parseKeywordsRecord(line);
            else if (recordType == "EXPDTA") parseExpDataRecord(line);
            else if (recordType == "REVDAT") parseRevDataRecord(line);
            else if (recordType == "AUTHOR") parseAuthorRecord(line);
            else if (recordType == "JRNL  ") parseJournalRecord(line);
            else if (recordType == "REMARK") parseRemarkRecord(line);
            else if (recordType == "DBREF ") parseDBRef(line);
            else if (recordType == "SEQADV") parseSequenceAdv(line);
            else if (recordType == "SEQRES") parseSequenceRes(line);
            else if (recordType == "FORMUL") parseFormula(line);
            else if (recordType == "HELIX ") parseHelix(line);
            else if (recordType == "SSBOND") parseSSBond(line);
            else if (recordType == "CRYST1") parseCryst1(line);
            else if (recordType == "ORIGX1") parseOriginX1(line);
            else if (recordType == "ORIGX2") parseOriginX2(line);
            else if (recordType == "ORIGX3") parseOriginX3(line);
            else if (recordType == "SCALE1") parseScale1(line);
            else if (recordType == "SCALE2") parseScale2(line);
            else if (recordType == "SCALE3") parseScale3(line);
            else if (recordType == "ATOM  ") parseAtom(line);
            else if (recordType == "TER   ") parseTer(line);
            else if (recordType == "HETATM") parseHetAtom(line);
            else if (recordType == "CONECT") parseConnection(line);
            else if (recordType == "MASTER") parseMaster(line);
            else if (recordType == "END   ") parseEnd(line);
            else
                printf("Unsupported record type '%s'\n", recordType.c_str());

        }
        inputStream.close();
    }

    std::vector<PDBFileAtom> atoms;
};

#endif //PDB_FORMAT_HPP