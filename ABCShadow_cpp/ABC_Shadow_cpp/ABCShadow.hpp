//
//  ABCShadow.hpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 13/01/2020.
//  Copyright © 2020 Quentin. All rights reserved.
//

#ifndef ABCShadow_hpp
#define ABCShadow_hpp

#include <iostream>
#include <fstream>
#include "MCMCSim.hpp"
#include "PottsModel.hpp"
#include "RandomGen.hpp"
#include "Stats.hpp"

using std::endl;
using std::ofstream;
using std::cerr;
using std::exp;
using std::cout;

class ABCShadow {
  
private:
    MCMCSim* mcmcSim;
    PottsModel* model;
    RandomGen* rGen;

    const Stats yObs;
    Stats theta0;
    Stats delta;
    vector<int> randomMask;
    int iter, n, samplerIt, samplerBy, samplerBurnin;
    double minBound, maxBound;
    bool gibbsSampler;
    
    vector<Stats> chain;
    
    Stats computeShadowChain(const Stats &theta);
    Stats sampleFromGibbs();
    Stats sampleFromMH();
    Stats getCandidate(const Stats &theta);
    
    double computeDensityRatio(const Stats &oldTheta,
                               const Stats &candidateTheta,
                               const Stats &ySim);

public:
    ABCShadow(MCMCSim* mcmcSim,
              PottsModel* model,
              RandomGen* rGen,
              Stats yObs,
              Stats theta0,
              Stats delta,
              vector<int> randomMask,
              int iter,
              int n,
              int samplerIt,
              int samplerBy=1,
              int samplerBurnin=0,
              int minBound=-100,
              int maxBound=100,
              bool gibbsSampler=true);
    
    void runABCShadow();
    void saveChain(ofstream &outputfile);
};

#endif /* ABCShadow_hpp */
