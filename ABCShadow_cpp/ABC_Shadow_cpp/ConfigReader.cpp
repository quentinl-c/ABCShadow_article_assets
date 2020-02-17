//
//  ConfigReader.cpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 15/01/2020.
//  Copyright © 2020 Quentin. All rights reserved.
//

#include "ConfigReader.hpp"

ConfigReader::ConfigReader() : sizeIn(12),
                               sizeOut(8),
                               seed(2018),
                               iter(1002000),
                               n(200),
                               samplerIt(200),
                               simIter(1000),
                               simBy(1),
                               name("default"),
                               path("./conf.txt"),
                               theta0(),
                               thetaPerf(),
                               delta(0.01, 0.01, 0.01),
                               yObs(),
                               simObs(true),
                               randomMask({1, 1, 1}){}

void ConfigReader::setAttr(string attrName, string val) {
    if(attrName == "sizeIn"){
        sizeIn = stoi(val);
    } else if(attrName == "sizeOut") {
        sizeOut = stoi(val);
    } else if(attrName == "seed") {
        seed = stoi(val);
    } else if(attrName == "iter") {
        iter = stoi(val);
    } else if(attrName == "n") {
        n = stoi(val);
    } else if(attrName == "samplerIt") {
        samplerIt = stoi(val);
    } else if(attrName == "simIter") {
        simIter = stoi(val);
    } else if(attrName == "simBy") {
           simBy = stoi(val);
    } else if(attrName == "name") {
        name = val;
    } else if(attrName == "theta0") {
        theta0 = toStats(val.c_str());
    } else if(attrName == "thetaPerf") {
        thetaPerf = toStats(val.c_str());
    } else if(attrName == "delta") {
        delta = toStats(val.c_str());
    } else if(attrName == "yObs") {
        yObs = toStats(val.c_str());
        simObs = false;
    } else if(attrName == "randomMask") {
        randomMask = toVec(val.c_str());
    } else {
        cout << val << " discarded, " << attrName << " doesn't match any attribute." << endl;
    }
}


void ConfigReader::read() {
    string delimiter{":"};
    ifstream configfile(path);
    if (configfile.is_open()){
        string line;
        while(getline(configfile, line)){
            line.erase(std::remove_if(line.begin(), line.end(), isspace), line.end());
            if(line[0] == '#' || line.empty()){
                continue;
            }
            auto delimiterPos = line.find(delimiter);
            auto attrName = line.substr(0, delimiterPos);
            auto value = line.substr(delimiterPos + 1);
            setAttr(attrName, value);
        }
        configfile.close();
    } else {
        cerr << "Couldn't open config file for reading.\n";
    }
}


void ConfigReader::print() {
    cout << "======== CONFIGURATION ========" << endl;
    cout << "📄 size: " << sizeIn << "," << sizeOut << endl;
    cout << "📄 theta_0: " << theta0 << endl;
    cout << "📄 delta: " << delta << endl;
    cout << "📄 seed: " << seed << endl;
    cout << "📄 iter: " << iter << endl;
    cout << "📄 n: " << n << endl;
    cout << "📄 sampler_it: " << samplerIt << endl;
    cout << "📄 y_obs: " << yObs << endl;
    cout << "🎭 randomMask: " << randomMask[0] << ", " << randomMask[1] << ", " << randomMask[2] << endl;
    if(simObs) {
        cout << "⚠️ Simulated data " << endl;
        cout << "📄 simIter: " <<  simIter << endl;
        cout << "📄 thetaPerf: " << thetaPerf << endl;
    }

}

void ConfigReader::save(const string &path) {
    ofstream configfile;
    configfile.open(path, std::ios::out | std::ios::trunc);
    if (configfile.is_open()){
        configfile << "size: " << sizeIn << "," << sizeOut << endl;
        configfile << "theta_0: " << theta0 << endl;
        configfile << "delta: " << delta << endl;
        configfile << "seed: " << seed << endl;
        configfile << "iter: " << iter << endl;
        configfile << "n: " << n << endl;
        configfile << "sampler_it: " << samplerIt << endl;
        configfile << "y_obs: " << yObs << endl;
        configfile << "randomMask: " << randomMask[0] << ", " << randomMask[1] << ", " << randomMask[2] << endl;
        if(simObs) {
            configfile << "simBy: " << simBy << endl;
            configfile << "simIter: " <<  simIter << endl;
            configfile << "thetaPerf: " << thetaPerf << endl;
        }

        configfile.close();
    } else {
        cerr << "Couldn't open config file for reading.\n";
    }
    
}
void ConfigReader::setSizeIn(const int &s) {sizeIn = s;}
int ConfigReader::getSizeIn() {return sizeIn;}
void ConfigReader::setSizeOut(const int &s) {sizeOut = s;}
int ConfigReader::getSizeOut() {return sizeOut;}
int ConfigReader::getSeed() {return seed;}
int ConfigReader::getIter() {return iter;}
int ConfigReader::getN() {return n;}
int ConfigReader::getSamplerIt() {return samplerIt;}
int ConfigReader::getSimIter() {return simIter;}
int ConfigReader::getSimBy() {return simBy;}
vector<int> ConfigReader::getRandomMask() {return randomMask;}
void ConfigReader::setTheta0(Stats newTheta) {theta0 = newTheta;}
Stats ConfigReader::getTheta0() {return theta0;}
void ConfigReader::setThetaPerf(Stats newTheta) {thetaPerf = newTheta;}
Stats ConfigReader::getThetaPerf() {return thetaPerf;}
Stats ConfigReader::getDelta() {return delta;}
void ConfigReader::setYObs(const Stats &stat) {yObs = stat;}
Stats ConfigReader::getYObs() {return yObs;}
void ConfigReader::setSimobs(bool flag) {simObs = flag;}
bool ConfigReader::isSimObs() {return simObs;}

void ConfigReader::setPath(const string &path) {this->path = path;}
void  ConfigReader::setName(const string &name) {this->name = name;}
string ConfigReader::getName() {return name;}


Stats toStats(const char *input){
    string s = input;
    string delimiter = ",";
    
    size_t pos = 0;
    string num;
    vector<double> vec{};
    while ((pos = s.find(delimiter)) != string::npos) {
        num = s.substr(0, pos);
        vec.push_back(atof(num.c_str()));
        s.erase(0, pos + delimiter.length());
    }
    vec.push_back(atof(s.c_str()));
    if(vec.size() < 3) {
        cerr << "Not the right input" << endl;
        return Stats{};
    } else {
        return Stats{vec[0],vec[1], vec[2]};
    }
}

vector<int> toVec(const char *input){
    string s = input;
    string delimiter = ",";
    
    size_t pos = 0;
    string num;
    vector<int> vec{};
    while ((pos = s.find(delimiter)) != string::npos) {
        num = s.substr(0, pos);
        vec.push_back(atoi(num.c_str()));
        s.erase(0, pos + delimiter.length());
    }
    vec.push_back(atof(s.c_str()));
    if(vec.size() < 3) {
        cerr << "Not the right input" << endl;
        return vector<int>{1,1,1};
    } else {
        return vec;
    }
}



