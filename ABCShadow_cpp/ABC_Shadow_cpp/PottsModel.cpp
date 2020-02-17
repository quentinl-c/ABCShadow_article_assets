//
//  PottsModel.cpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 05/08/2019.
//  Copyright © 2019 Quentin. All rights reserved.
//

#include "PottsModel.hpp"


PottsModel::PottsModel() : params{}{}

PottsModel::PottsModel(double theta11, double theta12, double theta22): params{{theta11, theta12, theta22}} {
    
}

double PottsModel::getTheta11()  const {
    return params[0];
}

double PottsModel::getTheta12() const {
    return params[1];
}

double PottsModel::getTheta22() const {
    return params[2];
}

void PottsModel::setParams(Stats params) {
    this->params = params;
}

double PottsModel::evaluateFromStats(const Stats &stats) {
    return params.dot(stats);
}

void PottsModel::applyChangeStats(Stats &delta_stats, const State &newState, Node* node) {
    node->applyPottsChangeStats(delta_stats, newState);
}

/*
double PottsModel::evaluateFromGraph(GraphWrapper &g) {
    return params.dot(g.getInteractionStats());
}

double PottsModel::getDelta(GraphWrapper &g, Node n, NodeType newType) {
    Vector3d deltaStats = g.toggle(n, newType);
    return params.dot(deltaStats);
}*/

State PottsModel::getNewProposal(RandomGen &rGen) {
    
    int i = rGen.getUniformIntD();
    return static_cast<State>(i);
}

Stats PottsModel::getStats(GraphWrapper &g) {
    return g.getInteractionStats();
}

Stats PottsModel::getParams() {
    return params;
}

Stats PottsModel::applyGibbsProposal(Node* node, RandomGen &rGen) {
    Stats up{};
    State newState, oldState{node->getNodeState()};
    double upProb{0};

    node->getGibbsChangeStatistics(up);

    upProb = exp(params.dot(up));
    upProb = upProb / (1+ upProb);

 
    if(upProb > rGen.getUnifornRealD()) {
        newState = State::ENABLED;
    } else {
        newState = State::DISABLED;
    }
   
    node->updateNodeState(newState);
    
    if(newState == oldState) {
        return Stats();
    }else if (newState == State::ENABLED) {
        return up;
    } else {
        return -up;
    }
    
}
