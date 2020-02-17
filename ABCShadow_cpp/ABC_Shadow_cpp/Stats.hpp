//
//  Stats.hpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 18/10/2019.
//  Copyright © 2019 Quentin. All rights reserved.
//

#ifndef Stats_hpp
#define Stats_hpp

#include <iostream>
#include <array>

using std::array;

class Stats {

private:
    
    array<double, 3> vec;
    int size{3};

public:
    Stats();
    Stats(const Stats& s);
    Stats(const double s1, const double s2, const double s3);
    Stats(const array<double, 3> ar);

    ~Stats();

    void increment(const int &idx);
    void decrement(const int &idx);
    void set(const int &idx, double newVal);
    void reset();
    double dot(const Stats &s) const;
    double operator[](const int &idx) const;
    Stats& operator+=(const Stats& s);
    Stats& operator/=(const double div);
    Stats& operator=(const Stats& s);
    Stats& operator-();
    friend Stats operator+(const Stats &s1, const Stats &s2);
    friend Stats operator-(const Stats &s1, const Stats &s2);
    friend Stats operator/(const Stats &s1, const double &div);
    friend std::ostream& operator<< (std::ostream &out, const Stats &stats);
    
    size_t getSize() const;
};

#endif /* Stats_hpp */
