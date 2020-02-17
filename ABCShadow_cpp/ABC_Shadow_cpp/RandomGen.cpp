//
//  RandomGen.cpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 12/01/2020.
//  Copyright © 2020 Quentin. All rights reserved.
//

#include "RandomGen.hpp"

RandomGen::RandomGen(unsigned int seed): seed{seed}, r_engine{seed}, unifIntD{0, 1}, unifRealD{} {}

mt19937 &RandomGen::getEngine() {
    return r_engine;
}

unsigned int RandomGen::getSeed() {
    return seed;
}

int RandomGen::getDiscreteInt(const vector<int> &probs) {
    discrete_distribution<> d(probs.begin(), probs.end());
    return d(r_engine);
}

int RandomGen::getUniformIntD() {
    return unifIntD(r_engine);
}

int RandomGen::getUniformIntD(int min, int max) {
    uniform_int_distribution<int> dist{min, max};
    return dist(r_engine);
}


double RandomGen::getUnifornRealD() {
    return unifRealD(r_engine);
}

double RandomGen::getUnifornRealD(double min, double max) {
    uniform_real_distribution<double> dist{min, max};
    return dist(r_engine);
}


