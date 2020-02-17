//
//  Stats.cpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 18/10/2019.
//  Copyright © 2019 Quentin. All rights reserved.
//

#include "Stats.hpp"

Stats::Stats(): vec{} {}

Stats::Stats(const array<double, 3> ar ): Stats() {vec = ar; }

// Copy constructor
Stats::Stats(const Stats &s): Stats(s.vec){}
        
Stats::Stats(const double s1, const double s2, const double s3): vec{} {
    vec[0] = s1;
    vec[1] = s2;
    vec[2] = s3;
}

Stats::~Stats() {}



void Stats::increment(const int &idx) {
    vec[idx] ++;
}

void Stats::decrement(const int &idx) {
    vec[idx] --;
}

void Stats::set(const int &idx, double newVal) {
    vec[idx] = newVal;
}

void Stats::reset() {
    for (int i = 0; i < size; i++) {
        vec[i] = 0;
    }
}

double Stats::dot(const Stats &s) const {
    double res{0};
    for(int i{0}; i < size; i++) {
        res += vec[i] * s[i];
    }
    return res;
}

double Stats::operator[](const int &idx) const {
    return vec[idx];
}

Stats& Stats::operator-() {
    for(int i{0}; i < size; i++){
        vec[i] = -1 * vec[i];
    }
    
    return *this;
}

Stats& Stats::operator=(const Stats &s) {
    size = s.size;
    vec = s.vec;
    return *this;
}
Stats& Stats::operator+=(const Stats &s) {
    for(int i{0}; i < size; i++){
        vec[i] += s[i];
    }
    
    return *this;
}

Stats& Stats::operator/=(const double div) {
    for(int i{0}; i < size; i++){
        vec[i] /= div;
    }
       
   return *this;
}

Stats operator+(const Stats &s1, const Stats &s2) {
    Stats res{};
    for(int i{0}; i < s1.size; i++) {
        res.vec[i] = s1[i] + s2[i];
    }
    return res;
}

Stats operator-(const Stats &s1, const Stats &s2) {
    Stats res{};
    for(int i{0}; i < s1.size; i++) {
        res.vec[i] = s1[i] - s2[i];
    }
    return res;
}

Stats operator/(const Stats &s1, const double &div) {
    Stats res{};
    for(int i{0}; i < s1.size; i++) {
        res.vec[i] = s1[i] / div;
    }
    return res;
}



std::ostream& operator<< (std::ostream &out, const Stats &stats) {
    out  << stats[0] << "," << stats[1] << "," << stats[2];
    return out;
}
