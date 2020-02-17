//
//  ABCShadow.cpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 13/01/2020.
//  Copyright © 2020 Quentin. All rights reserved.
//

#include "ABCShadow.hpp"

ABCShadow::ABCShadow(MCMCSim* mcmcSim,
                     PottsModel* model,
                     RandomGen* rGen,
                     Stats yObs,
                     Stats theta0,
                     Stats delta,
                     vector<int> randomMask,
                     int iter,
                     int n,
                     int samplerIt,
                     int samplerBy,
                     int samplerBurnin,
                     int minBound,
                     int maxBound,
                     bool gibbsSampler):
                     mcmcSim(mcmcSim),
                     model(model),
                     rGen(rGen),
                     yObs(yObs),
                     theta0(theta0),
                     delta(delta),
                     randomMask(randomMask),
                     iter(iter),
                     n(n),
                     samplerIt(samplerIt),
                     samplerBy(samplerBy),
                     samplerBurnin(samplerBurnin),
                     minBound(minBound),
                     maxBound(maxBound),
                     gibbsSampler(gibbsSampler),
                     chain(){}

/*
 =====================================================
 Private methods
 =====================================================
 */

Stats ABCShadow::computeShadowChain(const Stats &theta) {
    //cout << "input theta: " << theta << endl;
    Stats thetaRes = theta;
    Stats candidate, ySim;
    double alpha, prob;

    model->setParams(thetaRes);
    
    if(gibbsSampler) {
        ySim = sampleFromGibbs();
    } else {
        ySim = sampleFromMH();
    }

    //cout << "ysim: " << ySim << endl;

    for(int i=0; i < n; i++) {
        candidate = getCandidate(thetaRes);
        //cout << "candidate: " << candidate << endl;
        //cout << "theta res: " << thetaRes << endl;
        alpha = computeDensityRatio(thetaRes, candidate, ySim);
        prob = rGen->getUnifornRealD();
        //cout << "prob: " << prob << endl;
        if(alpha > prob) {
            thetaRes = candidate;
        }
    }
    //cout << "Output theta: " << thetaRes << endl;
    return thetaRes;
}

Stats ABCShadow::sampleFromGibbs() {
    //cout << "Auxiliary variable sim: theta-> " << model->getParams() << " samplerIt-> " << samplerIt << " burnin-> " << samplerBurnin << " by " << samplerBy << endl;
    vector<Stats> samples = mcmcSim->gibbsSim(*rGen, *model, samplerIt, samplerBy, samplerBurnin);
    Stats avg{};
    int t = 1;
    for (auto el : samples) {
        avg = avg + (el - avg)/t;
        t +=1 ;
    }
    mcmcSim->reset();
    return avg;
}

Stats ABCShadow::sampleFromMH() {
    vector<Stats> samples = mcmcSim->mhSim(*rGen, *model, samplerIt, samplerBy, samplerBurnin);
    Stats avg{};
    int t = 1;
    for (auto el : samples) {
        avg = avg + (el - avg)/t;
        t +=1 ;
    }
    mcmcSim->reset();
    return avg;
}

Stats ABCShadow::getCandidate(const Stats &theta){
    //int index = rGen->getUniformIntD(0, 2);
    int index = rGen->getDiscreteInt(randomMask);

    Stats candidate = theta;
    double oldVal = candidate[index];
    double newVal = rGen->getUnifornRealD(oldVal - delta[index] / 2, oldVal + delta[index] / 2);

    newVal = (newVal > maxBound) ? maxBound : newVal;
    newVal = (newVal < minBound) ? minBound : newVal;
    candidate.set(index, newVal);

    return candidate;
}


double ABCShadow::computeDensityRatio(const Stats &oldTheta,
                                      const Stats &candidateTheta,
                                      const Stats &ySim) {
    double p1, p2, q1, q2, ratio;

    p1 = candidateTheta.dot(yObs);
    q2 = candidateTheta.dot(ySim);
    
    p2 = oldTheta.dot(yObs);
    q1 = oldTheta.dot(ySim);

    ratio = (exp(p1 - p2)) * (exp(q1 - q2));
    ratio = (ratio >= 1) ? 1 : ratio;
    //cout << "ratio: " << ratio << endl;
    return ratio;
}


/*
 =====================================================
 Public methods
 =====================================================
 */
void ABCShadow::runABCShadow() {
    cout << "Run the ABC" <<endl;
    
    Stats theta = theta0;
    chain.push_back(theta0);
    chain.clear();
    //cout << "size " << chain.size() << endl;
    for(int i=0; i< iter-1; i++) {
        //cout << "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::" << endl;
        theta = computeShadowChain(theta);
        /*if(i % 1000 == 0) {
            cout <<  i << " iters - theta :" << theta << "\n";
        }*/
        chain.push_back(theta);
    }
}

void ABCShadow::saveChain(ofstream &outputfile) {
    if( !outputfile ) {
      cerr << "Error: file could not be opened" << endl;
      exit(1);
    }
    //cout << "size " << chain.size() << endl;
    for(auto const &v : chain) {
        outputfile << v << endl;
    }
}
