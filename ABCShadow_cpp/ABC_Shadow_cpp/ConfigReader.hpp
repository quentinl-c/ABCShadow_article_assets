//
//  ConfigReader.hpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 15/01/2020.
//  Copyright © 2020 Quentin. All rights reserved.
//

#ifndef ConfigReader_hpp
#define ConfigReader_hpp

#include <iostream>
#include <fstream>
#include <vector>
#include "Stats.hpp"
#include <algorithm>

using std::string;
using std::ifstream;
using std::ofstream;
using std::stoi;
using std::cout;
using std::cerr;
using std::endl;
using std::vector;

Stats toStats(const char *input);
vector<int> toVec(const char *input);

class ConfigReader {
    
private:
    int sizeIn, sizeOut, seed, iter, n, samplerIt, simIter, simBy;
    string name, path;
    Stats theta0, thetaPerf, delta, yObs;
    bool simObs;
    vector<int> randomMask ;
    
    void setAttr(string attrName, string val);
public:
    ConfigReader();
    
    void read();
    void print();
    void save(const string &path);
    
    void setSizeIn(const int &s);
    int getSizeIn();
    void setSizeOut(const int &s);
    int getSizeOut();
    int getSeed();
    int getIter();
    int getN();
    int getSamplerIt();
    int getSimIter();
    int getSimBy();
    vector<int> getRandomMask();
    void setPath(const string &path);
    void setTheta0(Stats newTheta);
    Stats getTheta0();
    void setThetaPerf(Stats newTheta);
    Stats getThetaPerf();
    Stats getDelta();
    void setYObs(const Stats &stat);
    Stats getYObs();
    void setSimobs(bool flag);
    bool isSimObs();
    string getName();
    void setName(const string &name);
};
#endif /* ConfigReader_hpp */
