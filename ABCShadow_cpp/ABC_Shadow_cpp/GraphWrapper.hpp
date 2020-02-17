//
//  GraphWrapper.hpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 02/08/2019.
//  Copyright © 2019 Quentin. All rights reserved.
//

#ifndef GraphWrapper_hpp
#define GraphWrapper_hpp

#include <iostream>
#include <vector>
#include <set>
#include <utility>
#include "Stats.hpp"
#include "Node.hpp"

using std::vector;


class GraphWrapper {
private:
    int inSize;
    int outSize;
    vector<Node> nodeList;

public:

    GraphWrapper(const int inSize, const int outSize);
    ~GraphWrapper();
    void empty();
    vector<Node> &getNodes();
    Stats getInteractionStats();
};
#endif /* GraphWrapper_hpp */
