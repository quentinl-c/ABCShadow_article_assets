//
//  main.cpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 02/08/2019.
//  Copyright � 2019 Quentin. All rights reserved.
//

#include <iostream>
#include "GraphWrapper.hpp"
#include <numeric>
#include "PottsModel.hpp"
#include "MCMCSim.hpp"
#include <chrono> // for std::chrono functions
#include <fstream>
#include <map>
#include "RandomGen.hpp"
#include "ABCShadow.hpp"
#include "ConfigReader.hpp"

using std::string;
using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;

using clock_time = std::chrono::high_resolution_clock;
using second_t = std::chrono::duration<double, std::ratio<1> >;

/*
 Utils
 */
void saveSimTraces(ofstream &outputfile, const vector<Stats> &traces);
Stats avgTraces(const vector<Stats> &traces);

int main(int argc, const char * argv[]) {
    
    string action;
    string configPath;
    ConfigReader confReader{};

    if(argc >= 3) {
        action = argv[1];
        configPath = argv[2];
        confReader.setPath(configPath);
        confReader.read();
    } else {
        cerr << "Usage : " << argv[0] << " action action configPath [name] [yObs/theta] [inSize] [outSize]" << endl;
        exit(-1);
    }
     
    if(argc >= 7) {
        cout << "Real data: " << endl;
        string name = argv[3];
        cout << "name: " << name << endl;
        confReader.setName(name);
        int inSize = std::atoi(argv[5]);
        confReader.setSizeIn(inSize);
        int outSize = std::atoi(argv[6]);
        confReader.setSizeOut(outSize);
        cout << "size: " << inSize << "," << outSize << endl;
        Stats stat = toStats(argv[4]);
        cout << "input vec: " << stat << endl;
        
        if (action == "abc_estim") {
            confReader.setYObs(stat);
            confReader.setSimobs(false);
        } else if (action == "mh_sim" || action == "gibbs_sim") {
            confReader.setThetaPerf(stat);
        }
    }
    

    GraphWrapper* g = new GraphWrapper(confReader.getSizeIn(), confReader.getSizeOut());
    PottsModel* model = new PottsModel();
    RandomGen* rGen = new RandomGen(confReader.getSeed());
    MCMCSim* mcmc = new MCMCSim(g);

    ofstream outputfile;
    outputfile.open(confReader.getName() + "_traces.csv", std::ios::out | std::ios::trunc);
    if( !outputfile) {
        cout << "Ça marche pas" << endl;
        cerr << "Error: output file could not be opened" << endl;
        exit(-1);
    }

    std::chrono::time_point<clock_time> time_start{};

    if (action == "abc_estim") {
        if(confReader.isSimObs()) {
            time_start = clock_time::now();
            cout << "Simulate the observation : " << confReader.getThetaPerf() << endl;
            model->setParams(confReader.getThetaPerf());
            vector<Stats> res = mcmc->gibbsSim(*rGen, *model, confReader.getSimIter());
            Stats yObs = avgTraces(res);
            confReader.setYObs(yObs);
            cout << "📊 End Sim generated observation : " << yObs << " in " << std::chrono::duration_cast<second_t>(clock_time::now() - time_start).count() << " s." << endl;
        }

        confReader.print();
        confReader.save(confReader.getName() + "_config.txt");
        time_start = clock_time::now();
        cout << "ABC Shadow is running ...  🚀 ⏳" << endl;
        
        ABCShadow* abc = new ABCShadow(mcmc,
                                       model,
                                       rGen,
                                       confReader.getYObs(),
                                       confReader.getTheta0(),
                                       confReader.getDelta(),
                                       confReader.getRandomMask(),
                                       confReader.getIter(),
                                       confReader.getN(),
                                       confReader.getSamplerIt());
        abc->runABCShadow();
        abc->saveChain(outputfile);
        double duration = std::chrono::duration_cast<second_t>(clock_time::now() - time_start).count();
        cout << "🎉 Execution has been completed  in " << duration << " s." << endl;
        
        outputfile.close();
    } else if (action == "mh_sim") {
        time_start = clock_time::now();
        model->setParams(confReader.getThetaPerf());
        vector<Stats> res = mcmc->mhSim(*rGen, *model, confReader.getSimIter(), confReader.getSimBy());
        saveSimTraces(outputfile, res);
        confReader.save(confReader.getName() + "_config_mh_sim.txt");
        Stats yObs = avgTraces(res);
        cout << "📊 End Sim generated observation : " << yObs << " in " << std::chrono::duration_cast<second_t>(clock_time::now() - time_start).count() << " s." << endl;
    } else if (action == "gibbs_sim") {
        time_start = clock_time::now();
        model->setParams(confReader.getThetaPerf());
        vector<Stats> res = mcmc->gibbsSim(*rGen, *model, confReader.getSimIter(), confReader.getSimBy());
        saveSimTraces(outputfile, res);
        confReader.save(confReader.getName() + "_config_gibbs_sim.txt");
        Stats yObs = avgTraces(res);
        outputfile.close();
        cout << "📊 End Sim generated observation : " << yObs << " in " << std::chrono::duration_cast<second_t>(clock_time::now() - time_start).count() << " s." << endl;
    } else {
        cerr << "No action matches to " << action << " [abc_estim] [mh_sim] [gibbs_sim]" << endl;
        exit(-2);
    }
   
    return 0;

}

void saveSimTraces(ofstream &outputfile, const vector<Stats> &traces) {
    if(outputfile.is_open()) {
        for (const auto &el : traces) {
            outputfile << el << endl;
        }
    } else {
        cerr << "Error: output file is not opened" << endl;
    }
}

Stats avgTraces(const vector<Stats> &traces) {
    Stats yObs{};
    int t {1};
    for (const auto &el : traces) {
        yObs = yObs + (el - yObs)/t;
        t +=1 ;
    }
    return yObs;
}
