//
//  Node.hpp
//  ABC_Shadow_cpp
//
//  Created by Quentin on 16/10/2019.
//  Copyright © 2019 Quentin. All rights reserved.
//

#ifndef Node_hpp
#define Node_hpp

#include <iostream>
#include <vector>
#include <memory>

#include "Stats.hpp"

using std::pair;
using std::vector;
using std::shared_ptr;

enum class State {
    DISABLED,
    ENABLED,
};

enum class NodeType {
    NIL, // 0
    INTRA, // 1
    INTER // 2
};

typedef pair<int, int> NodeName;

class Node {

private:
    NodeName nodeName;
    NodeType activationVal;
    shared_ptr<NodeType> label;
    vector<shared_ptr<NodeType>> neighbourhood;
    

public:
    Node();
    Node(int i, int j, NodeType activationVal);
    NodeName getName();
    void addNeighbour(const shared_ptr<NodeType> neighbourType);
    void addNeighbours(vector<Node> &neighbourhood);
    void setNodeType(const NodeType newLabel);
    void updateNodeState(const State newState);
    State getNodeState();
    shared_ptr<NodeType> getLabel();
    vector<shared_ptr<NodeType>> &getNeighbourhood();
    Stats getInteractionStats();
    void getGibbsChangeStatistics(Stats &stats);
    void applyPottsChangeStats(Stats &stats, const State &newState);
    
};
#endif /* Node_hpp */
