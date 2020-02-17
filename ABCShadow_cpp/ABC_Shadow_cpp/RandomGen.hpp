//
//  RandomGen.hpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 12/01/2020.
//  Copyright © 2020 Quentin. All rights reserved.
//

#ifndef RandomGen_hpp
#define RandomGen_hpp

#include <iostream>
#include <random>
#include <vector>

using std::vector;
using std::mt19937;
using  std::discrete_distribution;
using std::uniform_int_distribution;
using std::uniform_real_distribution;

class RandomGen {
private:
    unsigned int seed;
    mt19937 r_engine;
    uniform_int_distribution<int> unifIntD;
    uniform_real_distribution<double> unifRealD;

public:
    RandomGen(unsigned int seed);
    
    unsigned int getSeed();
    mt19937 &getEngine();
    int getDiscreteInt(const vector<int> &probs);
    int getUniformIntD();
    int getUniformIntD(int min, int max);
    double getUnifornRealD();
    double getUnifornRealD(double min, double max);
    
};

#endif /* RandomGen_hpp */
