//
//  MCMCSim.hpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 06/08/2019.
//  Copyright © 2019 Quentin. All rights reserved.
//

#ifndef MCMCSim_hpp
#define MCMCSim_hpp

#include <iostream>
#include "PottsModel.hpp"
#include "GraphWrapper.hpp"
#include "RandomGen.hpp"

using std::exp;


class MCMCSim {
private:
    GraphWrapper *graph;
    
public:
    MCMCSim(GraphWrapper *g);
    ~MCMCSim();
    
    void setGraph(GraphWrapper *new_g);
    
    void reset();
    vector<Stats> mhSim(RandomGen &rGen, PottsModel &model, const int iter_max, const int by=1, const int burnin=0);
    vector<Stats> gibbsSim(RandomGen &rGen, PottsModel &model, const int iter_max, const int by=1, const int burnin=0);
};

#endif /* MCMCSim_hpp */
