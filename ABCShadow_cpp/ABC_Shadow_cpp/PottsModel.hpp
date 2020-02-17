//
//  PottsModel.hpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 05/08/2019.
//  Copyright © 2019 Quentin. All rights reserved.
//

#ifndef PottsModel_hpp
#define PottsModel_hpp

#include <iostream>
#include <random>
#include "GraphWrapper.hpp"
#include "Stats.hpp"
#include "RandomGen.hpp"

using std::exp;

class PottsModel {
    
private:
    Stats params;

public:
    PottsModel();
    PottsModel(double theta11, double theta12, double theta22);
    
    double getTheta11() const;
    double getTheta12() const;
    double getTheta22() const;

    void setParams(Stats params);
    double evaluateFromStats(const Stats &stats);
    double evaluateFromGraph(GraphWrapper &g);
    double getDelta(GraphWrapper &g, Node n, NodeType newtype);
    void applyChangeStats(Stats &delta_stat, const State &newState, Node* node);
    Stats applyGibbsProposal(Node* node, RandomGen &rGen);
    State getNewProposal(RandomGen &rGen);
    Stats getStats(GraphWrapper &g);
    Stats getParams();
};

#endif /* PottsModel_hpp */
